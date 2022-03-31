import pdb
import config

import torch
import torch.nn as nn
import torch.nn.functional as F


class network_baseline(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(network_baseline, self).__init__()

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.name = 'baseline'

		print('... initialising {} network'.format(self.name))

		torch.manual_seed(51)
		torch.cuda.manual_seed_all(51)

		self.input_dim = input_dim
		self.output_dim = output_dim

		CNNcounter = 0

		if input_dim[0] != -1:

			CNNcounter += 1
			self.cnnHR = nn.Sequential(
							nn.Conv1d(input_dim[0],config._DIMENSION_EMBEDDINGS,2).float(),
							nn.BatchNorm1d(config._DIMENSION_EMBEDDINGS).float(),
							nn.ReLU(),
							nn.AdaptiveAvgPool1d((2)).float()
						)

		if input_dim[1] != -1:

			CNNcounter += 1
			self.cnnSTEPS = nn.Sequential(
							nn.Conv1d(input_dim[1],config._DIMENSION_EMBEDDINGS,2).float(),
							nn.BatchNorm1d(config._DIMENSION_EMBEDDINGS).float(),
							nn.ReLU(),
							nn.AdaptiveAvgPool1d((2)).float()
						)

		if input_dim[2] != -1:

			CNNcounter += 1
			self.cnnXYZ = nn.Sequential(
							nn.Conv1d(input_dim[2],config._DIMENSION_EMBEDDINGS,2).float(),
							nn.BatchNorm1d(config._DIMENSION_EMBEDDINGS).float(),
							nn.ReLU(),
							nn.AdaptiveAvgPool1d((2)).float()
						)

		self.layer_fullyConnected = nn.Sequential(
			nn.Dropout(p=0.3),
			nn.Linear((2*config._DIMENSION_EMBEDDINGS)*CNNcounter, config._DIMENSION_FC_INNERLAYERS, bias=True).float(),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(config._DIMENSION_FC_INNERLAYERS, output_dim, bias=True).float()
		)

		print('[NETWORK SUMMARY]')
		print(self)
		print('Network initialised successfully')


	def forward(self, X_hr, X_steps, X_xyz):

		if len(X_hr) != 0: 

			samplesInCurrentBatch = X_hr.shape[0]

			output_hr_cnn = self.cnnHR(X_hr)
			output_hr_cnn = output_hr_cnn.reshape(samplesInCurrentBatch, -1)

		if len(X_steps) != 0: 

			samplesInCurrentBatch = X_steps.shape[0]

			output_steps_cnn = self.cnnSTEPS(X_steps)
			output_steps_cnn = output_steps_cnn.reshape(samplesInCurrentBatch, -1)

		if len(X_xyz) != 0: 

			samplesInCurrentBatch = X_xyz.shape[0]

			output_xyz_cnn = self.cnnXYZ(X_xyz)
			output_xyz_cnn = output_xyz_cnn.reshape(samplesInCurrentBatch, -1)

		if (len(X_hr) != 0) and (len(X_steps) == 0) and (len(X_xyz) == 0): embeddedFeatures = output_hr_cnn
		elif (len(X_hr) == 0) and (len(X_steps) != 0) and (len(X_xyz) == 0): embeddedFeatures = output_steps_cnn
		elif (len(X_hr) == 0) and (len(X_steps) == 0) and (len(X_xyz) != 0): embeddedFeatures = output_xyz_cnn
		elif (len(X_hr) != 0) and (len(X_steps) != 0) and (len(X_xyz) == 0): embeddedFeatures = torch.cat((output_hr_cnn, output_steps_cnn),1)
		elif (len(X_hr) != 0) and (len(X_steps) == 0) and (len(X_xyz) != 0): embeddedFeatures = torch.cat((output_hr_cnn, output_xyz_cnn),1)
		elif (len(X_hr) == 0) and (len(X_steps) != 0) and (len(X_xyz) != 0): embeddedFeatures = torch.cat((output_steps_cnn, output_xyz_cnn),1)
		elif (len(X_hr) != 0) and (len(X_steps) != 0) and (len(X_xyz) != 0): embeddedFeatures = torch.cat((output_hr_cnn, output_steps_cnn, output_xyz_cnn),1)

		y_hat = self.layer_fullyConnected(embeddedFeatures)

		return y_hat

