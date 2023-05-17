from matchms.importing import load_from_msp
import numpy as np
import os
import random
import pickle
import glob
import time
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
import matchms
from matchms import Spectrum

import matplotlib.pyplot as plt
import warnings

from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn as nn
from torch_geometric.data import DataLoader

MAX_INTENSITY = 999.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def spectrum_preperation(tensor, max_intensity):
    # spectrum preparation
    # input:
    # tensor: (1, len_of_spectrum)
    # max_intensity: float value of highest value peak
    # return:
    # spectrum object from msml
    spectrum = tensor.detach().numpy()
    spectrum = spectrum / spectrum.max() * max_intensity

    position = np.nonzero(spectrum)[0]
    intensity = spectrum[position]

    spectrum = Spectrum(mz=position.astype(float),
                    intensities=intensity)

    return spectrum
    
def load_each_model(path, regex):
    # return path to all models in given folder
    absolute_folder_path = os.path.abspath(path)
    pt_files = glob.glob(os.path.join(absolute_folder_path, '*.pt'))
    return pt_files

def find_validation_files(folder_path, regex):
    # Given a folder path, returns a list of all files with a 'validation<digits>.pkl'
    
    absolute_folder_path = os.path.abspath(folder_path)
    validation_files = []
    regex = regex
    for filename in os.listdir(absolute_folder_path):
        if re.match(regex, filename):
            validation_files.append(os.path.join(absolute_folder_path, filename))
    return validation_files

def give_model_epoch(model_path):
    # model should have in format name ..path/number/pt
    file_name = os.path.basename(model_path)
    file_name_without_ext = os.path.splitext(model_path)[0]

    last_num_str = ''
    for c in reversed(file_name_without_ext):
        if c.isdigit():
            last_num_str = c + last_num_str
        else:
            break
    return last_num_str
    

def load_models_and_predict(models_path, model_dir, model_category, loader):
    # for each model make prediction and save it in directory
    for model_path in models_path:
        model = model_category
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        pred_concat = torch.tensor([])
        true_concat = torch.tensor([])

        with torch.no_grad():
            for batch in loader:
                # Use GPU
                batch.to(device)       
                # Passing the node features and the connection info
                pred = model(batch.x.float(), batch.edge_index, batch.edge_attr, batch.molecular_weight, batch.batch) 

                true_concat = torch.cat((true_concat, batch.y), 0)
                pred_concat = torch.cat((pred_concat, pred), 0)
            
            tensor_prediction = {'true': true_concat, 'pred': pred_concat}

            with open(model_dir +'/validation' + give_model_epoch(model_path) + '.pkl', 'wb') as f:
                pickle.dump(tensor_prediction, f)
        
        # save ram memory
        model = None
        del model
        true_concat = None
        del true_concat
        pred_concat = None
        del pred_concat
        tensor_prediction = None
        del tensor_prediction
def compute_statistics_for_validation(validations_paths):
    # compute dot product and stein dot product on files we save in load_models_and_predict

    result = dict()
    for validation in validations_paths:

        with open(validation, 'rb') as f:
            loaded_validation = pickle.load(f)
        
        true_concat = loaded_validation['true']
        pred_concat = loaded_validation['pred']


        sim_dp = matchms.similarity.CosineGreedy(mz_power=1., intensity_power=.5)
        sim_sdp = matchms.similarity.CosineGreedy(mz_power=3., intensity_power=.6)

        dp =[]
        sdp = []
     
        for true_tensor, pred_tensor in zip(true_concat, pred_concat):
            true_spectrum = spectrum_preperation(true_tensor, MAX_INTENSITY)

            pred_spectrum = spectrum_preperation(pred_tensor, MAX_INTENSITY)

            dp.append(sim_dp.pair(true_spectrum, pred_spectrum))
            sdp.append(sim_sdp.pair(true_spectrum, pred_spectrum))

        dp = np.array([ float(s['score']) for s in dp ])
        sdp = np.array([ float(s['score']) for s in sdp ])
        result[validation] = (dp, sdp)
       
    return result
    
def get_validation_number(file_path):
    # function to number to sort outputs of models
    match = re.search(r'validation(\d+)\.', file_path)
    if match:
        return int(match.group(1))
    else:
        return None
        
def plot_convolve_mean(statistics_validation):
    sorted_files = sorted(statistics_validation.keys(), key=get_validation_number)
    for file_name in sorted_files:
        value = statistics_validation[file_name]
        dp = value[0]
        sdp = value[1]

        win = 50
        smooth_dp = np.convolve(dp, np.ones(win)/win, mode='valid')
        smooth_sdp = np.convolve(sdp, np.ones(win)/win, mode='valid')

        plt.title(f"Validation at epoch {get_validation_number(file_name)}")
        plt.plot(smooth_dp,label='dp')
        plt.plot(smooth_sdp,label='sdp')
        plt.legend()

        plt.show()


def plot_histograms(statistics_validation):
    sorted_files = sorted(statistics_validation.keys(), key=get_validation_number)
    for file_name in sorted_files:
        value = statistics_validation[file_name]
        dp = value[0]
        sdp = value[1]

        bins = 70
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.hist(dp,label='dp',bins=bins)
        plt.title('dp')
        plt.subplot(122)
        plt.hist(sdp,label='sdp',bins=bins)
        plt.title('sdp')
        plt.suptitle(f"Validation at epoch {get_validation_number(file_name)}")
        plt.show()

def convolved_mean(path_models, statistics_validation):
    value = statistics_validation[path_models + "/validation10.pkl"]
    value1 = statistics_validation[path_models + "/validation100.pkl"]
    value2 = statistics_validation[path_models + "/validation200.pkl"]
    value3 = statistics_validation[path_models + "/validation290.pkl"]

    win = 50
    smooth_dp = np.convolve(value[0], np.ones(win)/win, mode='valid')
    smooth_sdp = np.convolve(value[1], np.ones(win)/win, mode='valid')

    plt.figure(figsize=(10, 8))

    # plot 1
    plt.subplot(2, 2, 1)
    plt.plot(smooth_dp,label='dp')
    plt.plot(smooth_sdp,label='sdp')
    plt.title('Convolved mean at 10')
    plt.legend()

    # plot 2
    plt.subplot(2, 2, 2)
    smooth_dp = np.convolve(value1[0], np.ones(win)/win, mode='valid')
    smooth_sdp = np.convolve(value1[1], np.ones(win)/win, mode='valid')
    plt.plot(smooth_dp,label='dp')
    plt.plot(smooth_sdp,label='sdp')
    plt.title('Convolved mean at 100')
    plt.legend()

    # plot 3
    plt.subplot(2, 2, 3)
    smooth_dp = np.convolve(value2[0], np.ones(win)/win, mode='valid')
    smooth_sdp = np.convolve(value2[1], np.ones(win)/win, mode='valid')
    plt.plot(smooth_dp,label='dp')
    plt.plot(smooth_sdp,label='sdp')
    plt.title('Convolved mean at 200')
    plt.legend()

    # plot 4
    plt.subplot(2, 2, 4)
    smooth_dp = np.convolve(value3[0], np.ones(win)/win, mode='valid')
    smooth_sdp = np.convolve(value3[1], np.ones(win)/win, mode='valid')
    plt.plot(smooth_dp,label='dp')
    plt.plot(smooth_sdp,label='sdp')
    plt.title('Convolved mean at 300')
    plt.legend()

    plt.show()

def histograms_convolved(path_models, statistics_validation):
    value = statistics_validation[path_models + "/validation100.pkl"]
    value2 = statistics_validation[path_models + "/validation200.pkl"]
    value3 = statistics_validation[path_models + "/validation290.pkl"]

    bins = 70
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.hist(value[0], label='100', bins=bins, alpha=0.5)
    plt.hist(value2[0], label='200', bins=bins, alpha=0.5)
    plt.hist(value3[0], label='300', bins=bins, alpha=0.5)
    plt.title('DP at 100, 200 and 300 epoch')
    plt.legend()
    plt.subplot(122)
    plt.hist(value[1], label='100', bins=bins, alpha=0.5)
    plt.hist(value2[1], label='200', bins=bins, alpha=0.5)
    plt.hist(value3[1], label='300', bins=bins, alpha=0.5)
    plt.title('SDP at 100, 200 and 300 epoch')
    plt.legend()
    plt.show()
    
def print_mean_and_deviation(statistics_validation):
    for key, value in statistics_validation.items():
        dp = value[0]
        sdp = value[1]
        
        print(key)
        print(f" | DotProduct mean is: {np.mean(dp)} and standard deviation is {np.std(dp)}")
        print(f" | SteinDotProduct mean is: {np.mean(sdp)} and standard deviation is {np.std(sdp)}")
        print()

def plot_progres_bar(statistics_validation):
    sorted_files = sorted(statistics_validation.keys(), key=get_validation_number)

    mean_dps = [ np.mean(statistics_validation[key][0]) for key in sorted_files]
    std_dps = [ np.std(statistics_validation[key][0]) for key in sorted_files]
    mean_sdps = [ np.mean(statistics_validation[key][1]) for key in sorted_files]
    std_sdps = [ np.std(statistics_validation[key][1]) for key in sorted_files]

    # median_dps = [ np.median(dps[f]) for f in sorted(dps.keys()) ]

    plt.figure(figsize=(12,3))
    l = len(mean_dps)
    plt.subplot(121)
    plt.errorbar(x=range(l),y=mean_dps,yerr=std_dps,label='dp',linestyle='None', marker='.',capsize=3)
    plt.legend()
    plt.subplot(122)
    plt.errorbar(x=range(l),y=mean_sdps,yerr=std_sdps,label='sdp',color='orange',linestyle='None', marker='.',capsize=3)
    plt.legend()
    plt.show()

def plot_median(statistics_validation):
    sorted_files = sorted(statistics_validation.keys(), key=get_validation_number)

    dpss = np.stack([statistics_validation[key][0] for key in sorted_files])
    dpss = np.sort(dpss)
    bs = dpss.shape[1]
    dpss_median = np.median(dpss,axis=1)
    dpss_percentile = dpss[:,(int(bs*.1), int(bs*.9))].T

    plt.figure(figsize=(18,12))
    plt.grid()
    plt.ylim(0.5,1)
    plt.errorbar(x=range(dpss.shape[0]),
                y=dpss_median,yerr=np.abs(dpss_percentile-dpss_median),
                linestyle='None',marker='.',capsize=3)
    plt.show()
    
def plot_boxplot_outliers(statistics_validation):
    sorted_files = sorted(statistics_validation.keys(), key=get_validation_number)

    dpss = np.stack([statistics_validation[key][0] for key in sorted_files])
    dpss = np.sort(dpss)
    plt.figure(figsize=(18,12))
    plt.boxplot(dpss.T)
    plt.show()
    
def validate_dataset(loader, model):
    # return 
    # true_list: list of true spetrum object
    
    pred_list = []
    true_list = []
    
    sim_dp = matchms.similarity.CosineGreedy(mz_power=1., intensity_power=.5)
    sim_sdp = matchms.similarity.CosineGreedy(mz_power=3., intensity_power=.6)

    dp =[]
    sdp = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)       
            # Passing the node features and the connection info
            pred = model(batch.x.float(), batch.edge_index, batch.edge_attr, batch.molecular_weight, batch.batch)

            for true_tensor, pred_tensor in zip(batch.y, pred):
                true_spectrum = spectrum_preperation(true_tensor, MAX_INTENSITY)
                true_list.append(true_spectrum)

                pred_spectrum = spectrum_preperation(pred_tensor, MAX_INTENSITY)
                pred_list.append(pred_spectrum)

                dp.append(sim_dp.pair(true_spectrum, pred_spectrum))
                sdp.append(sim_sdp.pair(true_spectrum, pred_spectrum))


    dp = np.array([ float(s['score']) for s in dp ])
    sdp = np.array([ float(s['score']) for s in sdp ])
    return true_list, pred_list, dp, sdp
    
def measure_time(loader, model):
    # return 
    # true_list: list of true spetrum object
    batch_times = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)       
            # Passing the node features and the connection info
            start_time = time.time()

            pred = model(batch.x.float(), batch.edge_index, batch.edge_attr, batch.molecular_weight, batch.batch)

            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
        print(len(batch_times))
        print(batch_times)
        mean_batch_time = sum(batch_times) / len(batch_times)
        print(f"Mean time taken per batch: {mean_batch_time:.5f} seconds")  
        
        
        
        
        
        
        
        
        
        
        
        
        
