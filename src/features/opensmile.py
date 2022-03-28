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
import os, yaml, glob
from os.path import basename
from tqdm import tqdm

params = {}
with open('params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params['opensmile']

feature_set = params['featureSet']

# base directory for audio files
audio_base='./dist/wav'
output_base='./dist/features/opensmile'
wavs = glob.glob(f'{audio_base}/*.wav')
os.makedirs(output_base, exist_ok=True)
outfile_base = os.path.join(output_base, 'features')
if "prosody" in feature_set:
    outfile = f'{outfile_base}_lld.csv'
else:
    outfile = f'{outfile_base}.csv'
outfile_lld = f'{outfile_base}_lld.csv'
for filename in tqdm(wavs):
    cmd = f'SMILExtract -noconsoleoutput -C ./opensmile/config/{feature_set} -I {filename} -N {basename(filename)} -csvoutput {outfile} -lldcsvoutput {outfile_lld} -timestampcsvlld 0 -appendcsv 1 -appendcsvlld 1'
    os.system(cmd)