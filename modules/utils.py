import argparse, os, sys, platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from modules import AEC_builder

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def append_default_path(verbose=False):
    temp_path = adapt_path(os.getcwd())
    temp_path = temp_path.split('Deep_clustering_v02')[0]
    temp_path = os.path.join(temp_path, 'Deep_clustering_v02')

    path_module = os.path.join(temp_path, 'modules')
    sys.path.append(path_module) # path_module을 경로에 추가

    path_csv = os.path.join(temp_path, 'csv_files')
    sys.path.append(path_csv) # path_csv를 경로에 추가

    # path_scalogram = adapt_path(os.getcwd()).split('Coding')[0]
    # path_scalogram = os.path.join(path_scalogram, 'Coding', 'data', 'Scalogram')
    path_scalogram = '/home/yschoi/data/scalogram'
    sys.path.append(path_scalogram)

    if verbose:
        print(path_module)
        print(path_csv)
        print(path_scalogram)
    
    return path_module, path_csv, path_scalogram

def adapt_path(path):
    path_identifier = get_path_identifier()

    return path.replace('\\', path_identifier)

def get_path_identifier():
    if platform.system() == 'Windows':
        path_identifier = '\\'
    elif platform.system() == 'Linux':
        path_identifier = '/'

    return path_identifier

def get_ExperimentInfo():
    path_module, path_csv, path_scalogram = append_default_path()
    df_experiment = pd.read_excel(os.path.join(path_csv, 'AEC_experiments.xlsx'), index_col=0)
    return df_experiment

def getTupleFromStr(_str):
    width = _str.split(',')[0]
    height = _str.split(',')[1]
    
    width = int(width.split('(')[1])
    height = int(height.split(')')[0])
    
    return (width, height)


