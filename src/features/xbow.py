#!/usr/bin/env python
import os, yaml
from posixpath import basename, splitext
import pandas as pd
import numpy as np
import sys
from glob import glob


if __name__=="__main__":
    params = {}
    with open('params.yaml') as f:
        p = yaml.load(f, Loader=yaml.FullLoader)
        params = p['xbow']
        feature_set = p['opensmile']['featureSet']



    conf_attributes_map = {'ComParE_2016': 'nt1[129]', 'eGeMAPSv01b': 'nt1[22]', 'densenet121': 'nt1[1024]', 'resnet50': 'nt1[2048]', 'prosodyShs': 'nt1[3]'}

    # Paths
    input_folder = f'./dist/features/opensmile'
    output_folder = f'./dist/features/xbow/'

    feature_config = splitext(feature_set.split("/")[-1])[0]

    feature_file = f'{input_folder}/features_lld.csv'
    if not os.path.isfile(feature_file):
        feature_file = f'{input_folder}/features.csv'
    all_features = pd.read_csv(feature_file, delimiter=';', quotechar="'")
    attributes = conf_attributes_map[feature_config]
    result_dir = f'{output_folder}'
    os.makedirs(result_dir, exist_ok=True)
    all_features.to_csv(os.path.join(result_dir, 'tmp-features.csv'), sep=';', index=None)
    csize = params['csize']
    num_assignments = params['num_assignments']

    # for csize in params['csize']:
    #     for num_assignments in params['num_assignments']:
    output_dir = result_dir
    os.makedirs(output_dir, exist_ok=True)
    xbow_config = f'-i {os.path.join(result_dir, "tmp-features.csv")}  -o {os.path.join(output_dir, "features.csv")} -attributes {attributes}'
    xbow_config += f' -standardizeInput -size {csize} -a {num_assignments} -log -B {os.path.join(output_dir, "codebook")}'
    print(xbow_config)
    os.system('java -Xmx120G -jar ./openXBOW/openXBOW.jar -writeName -csvHeader ' + xbow_config)
    os.remove(os.path.join(result_dir, 'tmp-features.csv'))
    df = pd.read_csv(os.path.join(output_dir, "features.csv"), delimiter=";", quotechar="'")
    df = df.sort_values(by=[df.columns[0]])
    df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
