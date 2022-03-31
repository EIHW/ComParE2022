import os
import argparse
import json
import pdb

from datetime import datetime

from numpy.random import seed
from torch import save

import config

from module_harAGE import harAGEdataset
from network_modelArchitecture import network_baseline
import module_AIroutines as AI
from module_analyseResults import compute_UAR, generate_CSVforSubmission


def main(args):

	seed(13)

	# TODO: update output path 
	outputPath = './results'
	if not os.path.exists(outputPath): os.mkdir(outputPath)

	# models_outputPath = os.path.join(outputPath, 'Model_{}'.format(args.modelType))
	# resultsReports_outputPath = os.path.join(outputPath, 'Model_{}_ResultsReports'.format(args.modelType))
	models_outputPath = outputPath
	resultsReports_outputPath = outputPath

	if not os.path.exists(models_outputPath): os.mkdir(models_outputPath)
	if not os.path.exists(resultsReports_outputPath): os.mkdir(resultsReports_outputPath)

	timestampID = datetime.now().strftime("%Y%m%d_%H%M")

	current_experimentLabel = args.modelType + '_' + '+'.join(args.modality)
	# current_models_outputPath = timestampID + '_' + current_experimentLabel
	current_models_outputPath = current_experimentLabel

	if not os.path.exists(os.path.join(models_outputPath, current_models_outputPath)):
		os.mkdir(os.path.join(models_outputPath, current_models_outputPath))

	print('[MSG] Starting development stage ...')

	f = open(os.path.join(resultsReports_outputPath, current_models_outputPath, 'results.txt'), 'a')
	f.write('################# DEVEL STAGE ################# \n')
	f.close()

	# Dataloader initialisation
	dataset_train = harAGEdataset(args.modality, ['Train'])
	dataset_devel = harAGEdataset(args.modality, ['Devel'])

	# Model initialisation
	if args.modelType == 'baseline': model = network_baseline(dataset_train.featuresDim(), config._OUTPUT_CLASSES)

	# Training parameters initialisation
	trainParams = {
					'epochs': config._TRAINING_EPOCHS,
					'lr': config._LR,
					'patience': config._PATIENCE
				  }

	# Model training
	model, optimalEpochs, labelsSummary = AI.network_training(model, dataset_train, dataset_devel, trainParams, 'devel')

	# Results calculation
	UAR = compute_UAR(labelsSummary['y_hat'], labelsSummary['y'])

	f = open(os.path.join(resultsReports_outputPath, current_models_outputPath, 'results.txt'), 'a')
	f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' | Devel set | UAR {:5.2f} % | \n'.format(100*UAR))
	f.close()

	resultsDict = {}
	resultsDict['predictions'] = labelsSummary['y_hat'].tolist()
	resultsDict['GT'] = labelsSummary['y'].tolist()
	resultsDict['sampleName'] = labelsSummary['sampleName']

	with open(os.path.join(models_outputPath, current_models_outputPath, 'inferences.json'), 'w') as fp:
		json.dump(resultsDict, fp, sort_keys=True, indent=4)

	del dataset_train, dataset_devel, labelsSummary, resultsDict

	print('[MSG] Development stage completed')

	print('[MSG] Starting test stage ...')

	# Dataloader initialisation
	dataset_train = harAGEdataset(args.modality, ['Train', 'Devel'])
	dataset_test = harAGEdataset(args.modality, ['Test'])

	# Model re-initialisation
	model.__init__(model.input_dim, model.output_dim)

	# Update trainParams
	trainParams['epochs'] = optimalEpochs
	trainParams['patience'] = None

	# Model training
	model, _, labelsSummary = AI.network_training(model, dataset_train, dataset_test, trainParams, 'test')
	save(model.state_dict(), os.path.join(models_outputPath, current_models_outputPath, 'model.pth'))

	results_df = generate_CSVforSubmission(dataset_test.harAGEactivities, dataset_test.metadata, labelsSummary)
	results_df.to_csv(os.path.join(resultsReports_outputPath, current_models_outputPath, 'Team_{}_SubmissionID_{:02d}.csv'.format(args.teamName, args.submissionID)),sep=';',index=False)

	del dataset_train, dataset_test, labelsSummary

	print('[MSG] Test stage completed')

	print('<-- SUCCESS -->')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--modality', '-m', nargs='+', default=['hr', 'steps', 'xyz'])
	parser.add_argument('--modelType' , '-t', type=str, default='baseline')
	parser.add_argument('--teamName', '-tID', type=str, default='EIHW')
	parser.add_argument('--submissionID', '-sID', type=int, default=0)

	args = parser.parse_args()

	main(args)