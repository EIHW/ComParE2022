import os
import pdb

import config

import time
from datetime import datetime

from numpy import arange, argmin, argmax, zeros, concatenate, median
from random import shuffle

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from module_analyseResults import compute_UAR


def EarlyStopping(model, tmpDirectory, best_val_loss, val_loss, maxPatience, patienceCounter):

	stopTraining = False
		
	if not os.path.isdir(tmpDirectory + '/'):
		os.mkdir(tmpDirectory)

	if val_loss < best_val_loss:

		patienceCounter = 0

		best_val_loss = val_loss

		if os.path.exists(tmpDirectory + '/tmp_model.pth'):
			os.system('rm -f ' + tmpDirectory + '/tmp_model.pth')

		torch.save(model.state_dict(), tmpDirectory + '/tmp_model.pth')

	else:

		patienceCounter += 1

		if patienceCounter >= maxPatience:
			stopTraining = True

	return patienceCounter, best_val_loss, stopTraining


def train(model, trainData, optimizer, criterion):

	model.train()

	total_loss = 0 
	batchCounter = 0

	for X_hr, X_steps, X_xyz, y, _ in trainData:

		batchCounter += 1

		optimizer.zero_grad()

		if len(X_hr) != 0: X_hr = X_hr.to(model.device)
		if len(X_steps) != 0: X_steps = X_steps.to(model.device)
		if len(X_xyz) != 0: X_xyz = X_xyz.to(model.device)

		y = y.to(model.device)

		y_hat = model(X_hr, X_steps, X_xyz)

		loss = criterion(y_hat,y)

		loss.backward()

		torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], config._GRAD_CLIP_VALUE)

		total_loss += loss.item()

		optimizer.step()

		if len(X_hr) != 0: X_hr = X_hr.to('cpu')
		if len(X_steps) != 0: X_steps = X_steps.to('cpu')
		if len(X_xyz) != 0: X_xyz = X_xyz.to('cpu')

		y = y.to('cpu')
		y_hat = y_hat.to('cpu')

	return model, total_loss / batchCounter


def evaluate(model, evalData, criterion, processStage, return_output=False):

	model.eval()

	labelsSummary = {}

	total_loss = 0
	batchCounter = 0

	GT = []
	predictions = []
	fileIDs = []

	with torch.no_grad():

		for X_hr, X_steps, X_xyz, y, fileNames in evalData:

			if len(X_hr) != 0: X_hr = X_hr.to(model.device)
			if len(X_steps) != 0: X_steps = X_steps.to(model.device)
			if len(X_xyz) != 0: X_xyz = X_xyz.to(model.device)

			y = y.to(model.device)

			batchCounter += 1

			y_hat = model(X_hr, X_steps, X_xyz)

			if processStage == 'devel':
				loss = criterion(y_hat,y)
				total_loss += loss.item()
			elif processStage == 'test':
				total_loss = 0

			if len(X_hr) != 0: X_hr = X_hr.to('cpu')
			if len(X_steps) != 0: X_steps = X_steps.to('cpu')
			if len(X_xyz) != 0: X_xyz = X_xyz.to('cpu')

			y = y.to('cpu')
			y_hat = y_hat.to('cpu')

			if return_output:

				if len(GT) == 0:

					GT = y
					predictions = y_hat
					fileIDs = list(fileNames)

				else:

					GT = torch.cat((GT, y),0)
					predictions = torch.cat((predictions, y_hat),0)
					fileIDs = fileIDs + list(fileNames)

		if return_output:

			predictions = F.softmax(predictions, dim=1)
			predictions = torch.argmax(predictions, dim=1)

			predictions = predictions.numpy()
			GT = GT.numpy()

	if return_output:
		labelsSummary['y_hat'] = predictions
		labelsSummary['y'] = GT
		labelsSummary['sampleName'] = fileIDs

	return total_loss / batchCounter, labelsSummary


def network_training(model, dataset_train, dataset_eval, trainParams, processStage):

	best_val_loss = float("inf")
	patienceCounter = 0

	historicalLoss_train = zeros((trainParams['epochs']))
	historicalLoss_eval = zeros((trainParams['epochs'])) 
	historicalUAR_eval = zeros((trainParams['epochs']))

	model = model.to(model.device)
	optimizer = optim.Adam(model.parameters(), lr=trainParams['lr'])
	criterion = torch.nn.CrossEntropyLoss().to(model.device)

	trainingTimeID = int(time.time())

	# TODO: update tmp path 
	tmpDirectory = 'tmp_' + model.name + '_' + str(trainingTimeID)	

	for epoch in arange(trainParams['epochs']):

		epoch_start_time = time.time()

		trainData = DataLoader(dataset_train, batch_size=config._BATCH_SIZE, shuffle=True)
		evalData = DataLoader(dataset_eval, batch_size=int(3*config._BATCH_SIZE), shuffle=False)

		model, train_loss = train(model, trainData, optimizer, criterion)
		historicalLoss_train[epoch] = train_loss
		
		eval_loss, labelsSummary = evaluate(model, evalData, criterion, processStage, return_output=True)
		
		if processStage == 'devel':

			historicalLoss_eval[epoch] = eval_loss
			historicalUAR_eval[epoch] = compute_UAR(labelsSummary['y_hat'], labelsSummary['y'])
			print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' | Epoch {:3d} | time {:5.2f}s | training loss {:5.4f} | eval loss {:5.4f} | eval UAR {:5.2f} % |'.format(epoch, (time.time() - epoch_start_time), train_loss, eval_loss, 100*historicalUAR_eval[epoch]))

		elif processStage == 'test':

			print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' | Epoch {:3d} | time {:5.2f}s | training loss {:5.4f} |'.format(epoch, (time.time() - epoch_start_time), train_loss))

		if model.device == torch.device("cuda:0"): torch.cuda.empty_cache()	

		if (trainParams['patience'] is not None) and (epoch > 0):	
			
			patienceCounter, best_val_loss, stopTraining = EarlyStopping(model, tmpDirectory, best_val_loss, (1 - historicalUAR_eval[epoch]), trainParams['patience'], patienceCounter)
		
			if stopTraining: 
				model.load_state_dict(torch.load(tmpDirectory + '/tmp_model.pth'))
				os.system('rm -rf ' + tmpDirectory)
				break

	if os.path.exists(tmpDirectory): os.system('rm -rf ' + tmpDirectory)

	# Inferences on the evalData with the optimal model
	if processStage == 'devel':
		_, labelsSummary = evaluate(model, evalData, criterion, processStage, return_output=True)

	model = model.to('cpu')
	criterion = criterion.to('cpu')

	if processStage == 'devel': optimalEpochs = (argmax(historicalUAR_eval[:epoch + 1]) + 1)
	elif processStage == 'test':optimalEpochs = None

	return model, optimalEpochs, labelsSummary

