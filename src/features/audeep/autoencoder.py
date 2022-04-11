#!/usr/bin/env python
import yaml
import subprocess as sp
from glob import glob
from os.path import splitext, basename

if __name__=="__main__":
    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["audeep"]["autoencoder"]
    spectrogram_base = "./audeep_workspace/spectrograms"
    spectrograms = sorted(glob(f"{spectrogram_base}/*.nc"))

    output_base="./audeep_workspace/autoencoder"
    cmd = [
        "audeep", "t-rae", "train",
        "--num-epochs", str(params["num_epochs"]),
        "--batch-size", str(params["batch_size"]),
        "--learning-rate", str(params["learning_rate"]),
        "--keep-prob", str(params["keep_prob"]),
        "--cell", params["cell"],
        "--num-layers", str(params["num_layers"]),
        "--num-units", str(params["num_units"])]
    if params["bidirectional_encoder"]:
        cmd.append("--bidirectional-encoder")
    if params["bidirectional_decoder"]:
        cmd.append("--bidirectional-decoder")
    # Train one autoencoder on each type of spectrogram
    for spectrogram_file in spectrograms:

        # Base directory for the training run
        run_name=f"{output_base}/{splitext(basename(spectrogram_file))[0]}"

        # Directory for storing temporary files. The spectrograms are temporarily stored as TFRecords files, in order to
        # be able to leverage TensorFlows input queues. This substantially improves training speed at the cost of using
        # additional disk space.
        temp_dir=f"{run_name}/tmp"
        _cmd = cmd + ["--input", spectrogram_file, "--run-name", run_name, "--tempdir", temp_dir]
        print(_cmd)
        sp.run(_cmd)
