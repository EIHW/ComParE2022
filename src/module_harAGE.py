import os 
import pdb
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

import config

from torch.utils import data
from torch import zeros, tensor


def load_hr(sensorMeasurements, HRrest):

	hrData = sensorMeasurements['heartRate_BPM'].values
	hrData_debiased = np.subtract(hrData, HRrest['HRrest_median'].values)

	hr_velocity = np.diff(hrData_debiased, prepend=hrData_debiased[0])
	hr_acceleration = np.diff(hr_velocity, prepend=hr_velocity[0])

	X = zeros(3,len(hrData))

	X[0,:] = tensor(hrData_debiased).float()
	X[1,:] = tensor(hr_velocity).float()
	X[2,:] = tensor(hr_acceleration).float()

	return X


def load_steps(sensorMeasurements):

	steps_velocity = sensorMeasurements['steps_1stDerivative'].values
	steps_acceleration = sensorMeasurements['steps_2ndDerivative'].values

	X = zeros(2, len(steps_velocity))

	X[0,:] = tensor(steps_velocity).float()
	X[1,:] = tensor(steps_acceleration).float()

	return X


def read_xyz(measurements_df):

	xData = np.asarray(eval(''.join(measurements_df['accelerometer_milliG_xAxis'].values).replace(' ','').replace('][',',')))
	yData = np.asarray(eval(''.join(measurements_df['accelerometer_milliG_yAxis'].values).replace(' ','').replace('][',',')))
	zData = np.asarray(eval(''.join(measurements_df['accelerometer_milliG_zAxis'].values).replace(' ','').replace('][',',')))

	return xData, yData, zData


def load_xyz(sensorMeasurements):

	xData, yData, zData = read_xyz(sensorMeasurements[['accelerometer_milliG_xAxis', 'accelerometer_milliG_yAxis', 'accelerometer_milliG_zAxis']])

	xData_filtered = gaussian_filter1d(xData,1)	
	yData_filtered = gaussian_filter1d(yData,1)	
	zData_filtered = gaussian_filter1d(zData,1)

	xData_filtered_velocity = np.diff(xData_filtered, prepend=xData_filtered[0])
	xData_filtered_acceleration = np.diff(xData_filtered_velocity, prepend=xData_filtered_velocity[0])

	yData_filtered_velocity = np.diff(yData_filtered, prepend=yData_filtered[0])
	yData_filtered_acceleration = np.diff(yData_filtered_velocity, prepend=yData_filtered_velocity[0])

	zData_filtered_velocity = np.diff(zData_filtered, prepend=zData_filtered[0])
	zData_filtered_acceleration = np.diff(zData_filtered_velocity, prepend=zData_filtered_velocity[0])

	X = zeros(9,len(xData_filtered))

	X[0,:] = tensor(xData_filtered).float()
	X[1,:] = tensor(xData_filtered_velocity).float()
	X[2,:] = tensor(xData_filtered_acceleration).float()
	X[3,:] = tensor(yData_filtered).float()
	X[4,:] = tensor(yData_filtered_velocity).float()
	X[5,:] = tensor(yData_filtered_acceleration).float()
	X[6,:] = tensor(zData_filtered).float()
	X[7,:] = tensor(zData_filtered_velocity).float()
	X[8,:] = tensor(zData_filtered_acceleration).float()

	return X


class harAGEdataset(data.Dataset):

	def __init__(self, modalities, partitions):

		self.modalities = modalities

		self.harAGEactivities = {'lying': 0,
								 'sitting': 1,
								 'standing': 2,
								 'washingHands': 3,
								 'walking': 4,
								 'running': 5,
								 'stairsClimbing': 6,
								 'cycling': 7}

		# TODO: update data path 
		self.dataPath = './dist'

		self.samples = []; self.metadata = []; self.HRmetadata = [];

		for partition in partitions:

			if len(self.samples) == 0:

				self.samples = [os.path.join(partition, s) for s in sorted(os.listdir(os.path.join(self.dataPath, partition)))]
				self.metadata = pd.read_csv(os.path.join(self.dataPath, '{}_metadata.csv'.format(partition)), sep=';', header=0)
				self.HRmetadata = pd.read_csv(os.path.join(self.dataPath, '{}_HRrestSummary.csv'.format(partition)), sep=';', header=0)
			
			else:

				self.samples = self.samples + [os.path.join(partition, s) for s in sorted(os.listdir(os.path.join(self.dataPath, partition)))]
				self.metadata = pd.concat([self.metadata, pd.read_csv(os.path.join(self.dataPath, '{}_metadata.csv'.format(partition)), sep=';', header=0)], ignore_index=True)
				self.HRmetadata = pd.concat([self.HRmetadata, pd.read_csv(os.path.join(self.dataPath, '{}_HRrestSummary.csv'.format(partition)), sep=';', header=0)], ignore_index=True)	


	def __len__(self):

		return len(self.samples)


	def __getitem__(self, index):

		X_hr = []
		X_steps = []
		X_xyz = []

		sensorMeasurements = pd.read_csv(os.path.join(self.dataPath, self.samples[index]), sep=';', header=0)

		fileName, extension = os.path.splitext(self.samples[index].split('/')[1])

		sample_metadata = self.metadata[self.metadata['sampleName'] == fileName + extension]
		sample_HRrest = self.HRmetadata[self.HRmetadata['participantID'] == sample_metadata['participantID'].item()]

		if sample_metadata['activity'].item() in self.harAGEactivities.keys():
			y = self.harAGEactivities[sample_metadata['activity'].item()]
		else:
			y = -1		

		for modality in self.modalities:

			if modality == 'hr': X_hr = load_hr(sensorMeasurements, sample_HRrest)
			elif modality == 'steps': X_steps = load_steps(sensorMeasurements)
			elif modality == 'xyz': X_xyz = load_xyz(sensorMeasurements)

		return X_hr, X_steps, X_xyz, y, fileName


	def featuresDim(self):

		X_hr, X_steps, X_xyz, y, fileName = self.__getitem__(0)

		if (len(X_hr) != 0) and (len(X_steps) == 0) and (len(X_xyz) == 0): 	 return X_hr.shape[0], -1, -1
		elif (len(X_hr) == 0) and (len(X_steps) != 0) and (len(X_xyz) == 0): return -1, X_steps.shape[0], -1
		elif (len(X_hr) == 0) and (len(X_steps) == 0) and (len(X_xyz) != 0): return -1, -1, X_xyz.shape[0]
		elif (len(X_hr) == 0) and (len(X_steps) != 0) and (len(X_xyz) != 0): return -1, X_steps.shape[0], X_xyz.shape[0]
		elif (len(X_hr) != 0) and len(X_steps) == 0 and (len(X_xyz) != 0):   return X_hr.shape[0], -1, X_xyz.shape[0]
		elif (len(X_hr) != 0) and (len(X_steps) != 0) and (len(X_xyz) == 0): return X_hr.shape[0], X_steps.shape[0], -1
		else: return X_hr.shape[0], X_steps.shape[0], X_xyz.shape[0]

