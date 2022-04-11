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


# Uncomment for debugging of auDeep
# verbose_option=" --verbose --debug"

# Uncomment for debugging of shell script
# set -x;


if __name__ == "__main__":
    params = {}
    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["audeep"]["spectrograms"]

    workspace = "audeep_workspace"

    # base directory for audio files
    audio_base = "dist/wav/"

    parser = "audeep.backend.parsers.no_metadata.NoMetadataParser"

    spectrogram_base = f"{workspace}/spectrograms"

    for clip_below_value in params["clip_below_values"]:
        spectrogram_file = f"{spectrogram_base}/{clip_below_value}.nc"
        cmd = [
            "audeep",
            "preprocess",
            "--parser",
            parser,
            "--basedir",
            audio_base,
            "--output",
            spectrogram_file,
            "--window-width",
            str(params["window_width"]),
            "--window-overlap",
            str(params["window_overlap"]),
            "--fixed-length",
            str(params["fixed_length"]),
            "--center-fixed",
            "--clip-below",
            str(clip_below_value),
            "--mel-spectrum",
            str(params["mel_bands"]),
        ]
        print(cmd)
        sp.run(cmd)
