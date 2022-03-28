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
import pandas as pd
from glob import glob
from tqdm import tqdm




if __name__ == "__main__":
    with open("params/audeep.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        to_export = params["export"]
    workspace = "audeep_workspace"

    # base directory of autoencoder runs
    representation_base = f"{workspace}/representations"
    representation_file = f"{representation_base}/{to_export}.nc"

    feature_base = f"./dist/features/audeep/"
    
    cmd = [
        "audeep",
        "export", 
        "--input", representation_file,
        "--format", "CSV",
        "--labels-last",
        "--output", feature_base,
        "--name", "features"
    ]
    sp.run(cmd)

    # remove unnecessary columns
    df = pd.read_csv(os.path.join(feature_base, "features.csv"))
    df = df.drop(columns=["chunk_nr", "label_nominal", "label_numeric"])
    df.to_csv(os.path.join(feature_base, "features.csv"), index=False)

