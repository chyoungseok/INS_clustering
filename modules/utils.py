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

def get_ExperimentInfo(sheet_name):
    path_module, path_csv, path_scalogram = append_default_path()
    df_experiment = pd.read_excel(os.path.join(path_csv, 'AEC_experiments.xlsx'), index_col=0, sheet_name=sheet_name)
    return df_experiment

def getTupleFromStr(_str):
    width = _str.split(',')[0]
    height = _str.split(',')[1]
    
    width = int(width.split('(')[1])
    height = int(height.split(')')[0])
    
    return (width, height)

def detail_modification(ax):
    ax.set_ylabel('frequency[Hz]', fontsize=15);
    ax.set_xlabel('sleep period', fontsize=15);
    ax.invert_yaxis();
    ax.yaxis.set_ticks(range(0,16));
    ax.yaxis.set_ticklabels([0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8, 5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, 37.5], fontsize=15);
    ax.grid(False);


def plot_scalograms(scalograms, save_path=None):
    num_samples = len(scalograms)
    
    for i in range(num_samples):
        scalograms[i] = scalograms[i][np.newaxis, :, :, :]

    fig, axes = plt.subplots(1, num_samples, figsize=(10*num_samples,12));
    clim = (0.0, 1.5)

    for i in range(num_samples):
        ax = axes[i]
        im_input = ax.imshow(scalograms[i].reshape(2000,16).transpose(1,0), cmap='hot', aspect=100, clim=clim);
        ax.set_title('centroid_%d' % (i+1), fontsize=20); detail_modification(ax);
        fig.colorbar(im_input, shrink=0.5, ax=ax);  
        
    if save_path != None:
        plt.savefig(save_path)
   

    return fig, axes

def plot_scalograms(scalogram, fig, ax, title=None, save_path=None):
    
    scalogram = scalogram[np.newaxis, :, :, :]

    clim = (0.0, 1.5)

    im_input = ax.imshow(scalogram.reshape(2000,16).transpose(1,0), cmap='hot', aspect='auto', clim=clim);
    if not(title==None):
        ax.set_title(title, fontsize=20); 
    ax.tick_params(bottom=False, top=False, left=False, right=False);
    ax.set_xticklabels([]);
    ax.set_yticklabels([]);
    ax.invert_yaxis();
    
    # detail_modification(ax);
    # fig.colorbar(im_input, shrink=0.5, ax=ax);  