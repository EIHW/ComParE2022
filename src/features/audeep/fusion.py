#!/usr/bin/env python
# Copyright (C) 2020 Shahin Amiriparian, Michael Freitag, Maurice Gerczuk, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.
import os, yaml
import subprocess as sp
from glob import glob
from tqdm import tqdm



# ##########################################################
# # 4. Feature Fusion
# ##########################################################

# # File to which we write the fused representations
# fused_file="${output_base}/${taskName}-fused/representations.nc"

# # Fuse all learned representations
# if [ ! -f ${fused_file} ]; then
#     echo audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
#     echo
#     audeep${verbose_option} fuse --input ${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}*/*/representations.nc --output ${fused_file}
# fi

# ##########################################################
# # 5. Feature Export
# ##########################################################
# export_base="${workspace}/csv"
# if [ ! -f ${export_base} ]; then
# 	mkdir -p $export_base
# fi
# export_basename=""

# for clip_below_value in ${clip_below_values}; do
#     # Base directory for the training run
#     run_name="${output_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}${clip_below_value}/t-${num_layers}x${num_units}-${bidirectional_encoder_key}-${bidirectional_decoder_key}"

#     # The file containing the learned representations
#     representation_file="${run_name}/representations.nc"

#     # The filenames for the CSV feature sets
#     export_name="${export_basename}${clip_below_value}"

#     echo audeep export --input ${representation_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}
#     echo
#     audeep export --input ${representation_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}

#     # Copy features to the main csv directory of the subchallenge
#     if [ ! -f "${feature_base}${export_name}" ]; then
# 		mkdir -p ${feature_base}${export_name}
# 	fi
#     cp "${export_base}/partitions/train/${export_name}.csv" "${feature_base}${export_name}/train.csv"
#     cp "${export_base}/partitions/devel/${export_name}.csv" "${feature_base}${export_name}/devel.csv"
#     cp "${export_base}/partitions/test/${export_name}.csv" "${feature_base}${export_name}/test.csv"
# done

# export_name="${export_basename}fused"

# echo audeep export --input ${fused_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}
# echo
# audeep export --input ${fused_file} --format CSV --labels-last --output "${export_base}/partitions" --name ${export_name}

# # Copy features to the main csv directory of the subchallenge
# mkdir -p ${feature_base}/fused/
# cp "${export_base}/partitions/train/${export_name}.csv" "${feature_base}/fused/train.csv"
# cp "${export_base}/partitions/devel/${export_name}.csv" "${feature_base}/fused/devel.csv"
# cp "${export_base}/partitions/test/${export_name}.csv" "${feature_base}/fused/test.csv"

if __name__ == "__main__":
    params = {}
    with open("params/audeep.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["fusion"]

    workspace = "audeep_workspace"

    # base directory of autoencoder runs
    representation_base = f"{workspace}/representations"
    fusion_base = f"{workspace}/fusion"

    runs = map(os.path.normpath, sorted(glob(f"{run_base}/*/")))

    
    for run_name in runs:
        clip_below_value = run_name.split(os.sep)[-1]
        spectrogram_file=f"{spectrogram_base}/{clip_below_value}.nc"
        model_dir = f"{run_name}/logs"
        representation_file=f"{representation_base}/{clip_below_value}.nc"
        cmd = [
            "audeep",
            "t-rae", 
            "generate",
            "--model-dir", model_dir,
            "--input", spectrogram_file,
            "--output", representation_file
        ]
        print(cmd)
        sp.run(cmd)

    # for clip_below_value in params["clip_below_values"]:
    #     spectrogram_file = f"{spectrogram_base}/{clip_below_value}.nc"
    #     cmd = [
    #         "audeep",
    #         "preprocess",
    #         "--parser",
    #         parser,
    #         "--basedir",
    #         audio_base,
    #         "--output",
    #         spectrogram_file,
    #         "--window-width",
    #         str(params["window_width"]),
    #         "--window-overlap",
    #         str(params["window_overlap"]),
    #         "--fixed-length",
    #         str(params["fixed_length"]),
    #         "--center-fixed",
    #         "--clip-below",
    #         str(clip_below_value),
    #         "--mel-spectrum",
    #         str(params["mel_bands"]),
    #     ]
    #     print(cmd)
    #     sp.run(cmd)
