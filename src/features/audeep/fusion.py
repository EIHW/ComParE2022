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



if __name__ == "__main__":
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
