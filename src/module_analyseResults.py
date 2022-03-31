import pdb

from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore') 

def compute_UAR(y_hat, y):

	report = classification_report(y, y_hat, output_dict=True)

	UAR = report['macro avg']['recall']

	return UAR
	

def generate_CSVforSubmission(activitiesDict, metadata_df, labelsSummary):

	results_df = metadata_df[['sampleName', 'activity']]

	for inferredClass, sampleName in zip(labelsSummary['y_hat'], labelsSummary['sampleName']):

		ID = results_df[results_df['sampleName'] == sampleName + '.csv'].index.values[0]

		results_df['activity'].iloc[ID] = list(activitiesDict.keys())[inferredClass]

	return results_df

