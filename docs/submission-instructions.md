# Humbug submission template

This repository contains a fully functional baseline model for the Humbug challenge, and docker template to submit code which is automatically evaluated and scored without user access to the test data.
The code provides a starting point for creating a submission.

Please refer to [Mosquito event challenge README](./baseline-reproduction.md) for more detail on how to locally run the baseline, including feature extraction, model training, and evaluation over dev data.


## Submission guidelines
This template works both in Docker and in a local environment.
However, your submission will be evaluated in a Docker environment.
We recommend that you use Docker to develop and especially to test your code before submitting it.

The following folders and files *must* be used as described below:
- `requirements.txt`: Python packages required to run the model
  - Note that most common required packages are already included in the `Dockerfile` image. E.g. tensorflow, numpy, pandas, etc.
- `Dockerfile`: builds the container where the model runs
  - Update it if you require additional libraries that cannot be installed via `requirements.txt`
- `src`: store your model source code here (training, evaluation etc.)
- `src/predict.py`: prediction script; perform inference here
  - This script will be invoked with no parameters by the evaluation engine
- `data/audio/test`: input audio files; `src/predict.py` should read input data from here (see "Input format")
  - This is where the evaluation engine will place the test data for inference
- `data/predictions/test`: output predictions; `src/predict.py` should write output data here (see "Expected output format")
  - This is where the evaluation engine will expect the prediction outputs to be
- `models`: store your model files / checkpoints here; `src/predict.py` should load model / model weights from here
- `SUBMISSION_README.md`: add any additional information you'd like to share as part of your submission here

### Input format
Files stored in `data/audio/test` with the `.wav` extension.

Feel free to use any other path for development, but make sure the code that you submit reads input data from here.

### Expected output format
Files stored in `data/predictions/test` with names in the form of `baseline_<threshold>.csv`, where threshold is between `[0.1, 1.0]` in increments of `0.1`.
See `data/predictions/dev/*` and ["Scoring function"](./baseline-reproduction.md) for reference.

Feel free to write outputs to any other path for development, but make sure the code that you submit writes output data here.


## Docker setup
Before making a submission, test that your model runs locally with the provided train & dev data.

### Prerequisites
- Docker and NVIDIA prerequisites listed [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow#prerequisites)
- [Docker compose](https://docs.docker.com/compose/install/)

### Testing
```
docker-compose build
docker-compose up
```
After running the above, your predictions should appear in the `data/predictions/test` folder.
This is equivalent to the way your model will be evaluated.


## Creating a submission

### Prerequisites
- Make sure inference works on your device by following the instructions to under `Docker setup`
- Remove any unnecessary files, especially large ones e.g. unnecessary checkpoints

The machine that will be used to evaluate your submission will be an AWS `g4dn.xlarge`,
with 16GB of RAM and 16GB dedicated graphics memory. Feel free to tune your model to run optimally under these conditions.

### Creating the submission archive
The file structure described in the "Structure" should be preserved.
```
bash make_submission.sh
```
The above will build an archive with:
- `models`, where your model files / checkpoints should be stored
- `src`, where your model source code should be stored
- `requirements.txt`, where the list of additional required Python packages should be stored
- `docker-compose.yml` - this should not require any changes
- `SUBMISSION_README.md` - feel free to modify this file to add additional information
- `Dockerfile` - this should only require changes if you require additional libraries

Note: The provided baseline model and predict script are just a starting point.
Feel free to modify them to suit your needs.
Just make sure to output predictions in the same format as the provided example.


## Dev setup
We encourage developing in the dockerized environment.
However, you can choose to use a local environment for development.

### Prerequisites
- NVIDIA prerequisites listed [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow#prerequisites)
- Anaconda or Miniconda

### Setup
Use the provided conda environment template:
```
conda env create -f humbug.yml
conda activate humbug
```

### Evaluating locally
Use the provided `src/eval.py` script by passing it the path to the predictions and the ground truth.


## Pytorch setup
The baseline model we provide is Tensorflow-based.
However, you can choose to submit a Pytorch model.
You can switch to a Pytorch Docker setup by replacing `Dockerfile` and `requirements.txt` with the ones in the `.pytorch` folder:
```
cp .pytorch/Dockerfile .
cp .pytorch/requirements.txt .
```
Everything else about submitting and running locally still applies.

Note: there is no local conda environment template provided for Pytorch.
