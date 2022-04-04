# Baseline scripts for the ACM MM - ComParE 2022 Human Activity Recognition Sub-Challenge


The Human Activity Recognition Sub-Challenge (HARS) is a classification task that aims at recognising the activity participants were doing from the analysis of 20 sec of sensor measurements collected using a smartwatch. 

The Sub-Challenge baseline measure is the Unweighted Average Recall (UAR). 

This Sub-Challenge is based on the harAGE corpus, collected in the framework of the European Union's Horizon 2020 research and innovation programme under grant agreement No. 826506 (sustAGE).

The task is an 8-class classification problem, comprising the activities 'lying', 'sitting', 'standing', 'washingHands', 'walking', 'running', 'stairsClimbing', and 'cycling'. The baseline system proposes and end-to-end approach fusing the embedded representations learnt from the heart rate, pedometer, and accelerometer sensor measurements via concatenation. 

## Submission 
The participants are required to provice one prediction per 20 sec sequence of sensor measurements in the test set. Participants should submit a prediction file following the format: 

```
sampleName;activity
Test_0001.csv;washingHands
Test_0002.csv;walking
[...]
```

Each registered team has up to five (5) submissions for this Sub-Challenge.

To submit your csv files, login on the website: http://www5.cs.fau.de/compare/
with the credentials you received from the organisers.

More information on the challenge: http://www.compare.openaudio.eu/compare2022/

Please note that each team needs to submit at least one regular paper to the ComParE 2022 Special Session at ACM Multimedia 2022. 

This paper may include your methods and results for several Sub-Challenges.

## General Installation Instruction 
### Linux
If you have conda installed (either miniconda or anaconda), you can execute
```bash
conda env create -f .env-ymls/ComParE_harAGE_env.yml
```
to setup the virtual environment required for reproducing the baseline experiments. You can activate the `ComParE_harAGE` environment with
```bash
source ./activate ComParE_harAGE
```

## Data
Make sure you are on the correct branch regarding your chosen sub-challenge. Otherwise (with the virtual environment activated), checkout the desired branch. Move or copy the data from the challenge package into the project's root directory, such that the `dist` folder lies on the same level as `src`. The layout should look like this:
```
dist/
  |-- Devel/
  |-- Devel_HRrestSummary.csv
  |-- Devel_metadata.csv
  |-- Test/
  |-- Test_HRrestSummary.csv
  |-- Test_metadata.csv
  |-- Train/
  |-- Train_HRrestSummary.csv
  |-- Train_metadata.csv
```

## Source code
`src` contains the python scripts used to run the baseline pipeline, including a dataloader to easily load the harAGE data for mono- and multi-modal processing. 

## Reproducing the baseline

To reproduce the baseline results, participants can run the script

```bash
python src/reproduce_baseline.py

```

or to reproduce individual configurations:

```bash
python src/main_modelling.py -m [modalities] -t baseline -tID [teamID] -sID [submissionID]

```

The argument `modalities` determines which sensor measurements the dataloader should load (`hr`, `steps`, or `xyz`). Please check the `src/reproduce_baseline.py` script for an exact example of the API call. 

If everything goes well, a folder `results/` will be created, with the structure
```
results/
  |-- Model_baseline/
  |-- Model_baseline_ResultsReports/
```

The `Model_baseline_ResultsReports` directory stores the resulting `.pth` model, and a `.json` file summarising the ground truth and the inferred activities on the development set. The `Model_baseline_ResultsReports` directory stores a `.txt` file with the UAR scored on the development partition. 
