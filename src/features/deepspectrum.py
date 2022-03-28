#!/usr/bin/env python
# Copyright (C) 2020 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, Björn Schuller
#
# This file is part of DeepSpectrum.
#
# DeepSpectrum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSpectrum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DeepSpectrum. If not, see <http://www.gnu.org/licenses/>.
import os, yaml
from glob import glob
from tqdm import tqdm

verbose_option = "-v"

params = {}
with open("params.yaml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params["deepspectrum"]


# base directory for audio files
audio_base = f"./dist/wav"
output_base = f"./dist/features/deepspectrum"
if not os.path.exists(output_base):
    os.makedirs(output_base)


# plot mode: mel, spectrogram, chroma
mode = params["plotMode"]

# if mode == mel -> number of mel bands
nmel = params["melBands"]

# good choices can be "magma", "plasma", "viridis" or "cividis"
colour = params["colourMap"]

# if windoing
window_size = params["window_size"]
hop_size = params["hop_size"]


# DeepSpectrum supports the following pre-trained CNN networks: vgg16, vgg19, resnet50, inception_resnet_v2,
# xception, densenet121, densenet169, densenet201, mobilenet, mobilenet_v2, nasnet_large, nasnet_mobile, alexnet,
# squeezenet, googlenet.
network = params["extractionNetwork"]
outputFeat = os.path.join(output_base, f'features.csv')
# recommended feature layer for all pre-trained CNNs except for vgg16, vgg19, and alexnet: "avg_pool"
# recommended feature layer for vgg16, vgg19, and alexnet: "fc2"
if network == "vgg16" or network == "vgg19" or network == "alexnet":
    featureLayer = "fc2"
else:
    featureLayer = "avg_pool"
if mode == "mel":

    cmd = (
        "deepspectrum "
        + verbose_option
        + " features "
        + audio_base
        + " -o "
        + outputFeat
        + " -en "
        + network
        + " -fl "
        + featureLayer
        + " -m "
        + mode
        + " -nm "
        + str(nmel)
        + " -cm "
        + colour
        + " -nl -np 16"
    )
    if window_size is not None:
        cmd += " -t " + str(window_size) + " " + str(hop_size)
    print(cmd)
    os.system(cmd)
# subprocess.Popen(cmd, shell=True).wait()
elif mode == "spectrogram" or mode == "chroma":

    cmd = (
        "deepspectrum "
        + verbose_option
        + " features "
        + audio_base
        + " -o "
        + outputFeat
        + " -en "
        + network
        + " -fl "
        + featureLayer
        + " -m "
        + mode
        + " -cm "
        + colour
        + " -nl -np 16"
    )
    if window_size is not None:
        cmd += " -t " + str(window_size) + " " + str(hop_size)
    print(cmd)
    os.system(cmd)
