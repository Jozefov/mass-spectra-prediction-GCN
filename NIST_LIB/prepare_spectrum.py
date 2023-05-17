from matchms.importing import load_from_msp
import numpy as np
import os
import random
import pickle
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

matchms.set_matchms_logger_level("ERROR")
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
    
def validate_dataset(loader, model):
    # return 
    # true_list: list of true spetrum object

    pred_concat = torch.tensor([])
    true_concat = torch.tensor([])
    for batch in loader:
        # Use GPU
        batch.to(device)         
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.edge_attr, batch.molecular_weight, batch.batch) 

        true_concat = torch.cat((true_concat, batch.y), 0)
        pred_concat = torch.cat((pred_concat, pred), 0)
    
    true_list = []
    pred_list = []
    
    sim_dp = matchms.similarity.CosineGreedy(mz_power=1., intensity_power=.5)
    sim_sdp = matchms.similarity.CosineGreedy(mz_power=3., intensity_power=.6)

    dp =[]
    sdp = []
    for true_tensor, pred_tensor in zip(true_concat, pred_concat):
        true_spectrum = spectrum_preperation(true_tensor, MAX_INTENSITY)
        true_list.append(true_spectrum)

        pred_spectrum = spectrum_preperation(pred_tensor, MAX_INTENSITY)
        pred_list.append(pred_spectrum)

        dp.append(sim_dp.pair(true_spectrum, pred_spectrum))
        sdp.append(sim_sdp.pair(true_spectrum, pred_spectrum))



    dp = np.array([ float(s['score']) for s in dp ])
    sdp = np.array([ float(s['score']) for s in sdp ])
    return true_list, pred_list, dp, sdp
