from matchms.importing import load_from_msp
import numpy as np
import pandas as pd
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
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, GATConv,GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn as nn
from torch_geometric.data import DataLoader

EMBEDDING_SIZE = 80
NODE_FEATURES = 50
OUTPUT_SIZE = 1000
INTENSITY_POWER = 0.5
MASS_SHIFT = 5 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mask_prediction_by_mass(total_mass, raw_prediction, index_shift):
    # Zero out predictions to the right of the maximum possible mass.
    # input 
    # anchor_indices: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # data: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can heaviest fragment differ from weight of original molecule
    # 

    data = raw_prediction.type(torch.float64)
    
    total_mass = torch.round(total_mass).type(torch.int64)
    indices = torch.arange(data.shape[-1])[None, ...].to(device)

    right_of_total_mass = indices > (
            total_mass[..., None] +
            index_shift)
    return torch.where(right_of_total_mass, torch.zeros_like(data),
                        data)
                        
#############################
# TO CO HORE ALE PYTORCH
#############################
def scatter_by_anchor_indices(anchor_indices, data, index_shift):
    # reverse vector by anchor_indices and rest set to zero
    # input 
    # anchor_indices: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # data: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can heaviest fragment differ from weight of original molecule
    
    index_shift = index_shift
    anchor_indices = anchor_indices
    data = data.type(torch.float64)
    batch_size = data.shape[0]
    
    num_data_columns = data.shape[-1]
    indices = torch.arange(num_data_columns)[None, ...].to(device)
    shifted_indices = anchor_indices[..., None] - indices + index_shift
    valid_indices = shifted_indices >= 0

   

    batch_indices = torch.tile(
          torch.arange(batch_size)[..., None], [1, num_data_columns]).to(device)
    shifted_indices += batch_indices * num_data_columns

    shifted_indices = torch.reshape(shifted_indices, [-1])
    num_elements = data.shape[0] * data.shape[1]
    row_indices = torch.arange(num_elements).to(device)
    stacked_indices = torch.stack([row_indices, shifted_indices], axis=1)


    lower_batch_boundaries = torch.reshape(batch_indices * num_data_columns, [-1])
    upper_batch_boundaries = torch.reshape(((batch_indices + 1) * num_data_columns),
                                          [-1])

    valid_indices = torch.logical_and(shifted_indices >= lower_batch_boundaries,
                                     shifted_indices < upper_batch_boundaries)

    stacked_indices = stacked_indices[valid_indices]
    
    # num_elements[..., np.newaxis] v tf aj ked je shape (), tak vies urbit data[]
    # teraz to z napr. 6 da na [6]
    dense_shape = torch.tile(torch.tensor(num_elements)[..., None], [2]).type(torch.int32)

    scattering_matrix = torch.sparse.FloatTensor(stacked_indices.type(torch.int64).T,
                                                 torch.ones_like(stacked_indices[:, 0]).type(torch.float64),
                                                dense_shape.tolist())

    flattened_data = torch.reshape(data, [-1])[..., None]
    flattened_output = torch.sparse.mm(scattering_matrix, flattened_data)
    return torch.reshape(torch.transpose(flattened_output, 0, 1), [-1, num_data_columns])
    
#############################
# TO CO HORE ALE PYTORCH
#############################
def reverse_prediction(total_mass, raw_prediction, index_shift):
    # reverse vector by anchor_indices and rest set to zero and make preproessing
    # input 
    # total_mass: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # raw_prediction: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can heaviest fragment differ from weight of original molecule
    #     total_mass = feature_dict[fmap_constants.MOLECULE_WEIGHT][..., 0]
    
    total_mass = torch.round(total_mass).type(torch.int32)
    return scatter_by_anchor_indices(
        total_mass, raw_prediction, index_shift)
        
        
class SKIPblock(nn.Module):
    def __init__(self, in_features, hidden_features, USE_dropout=True, dropout_rate = 0.2):
        super().__init__()
        #only need to change shape of the residual if num_channels changes (i.e. in_c != out_c)
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]
        
        self.hidden1= nn.utils.weight_norm(nn.Linear(in_features, hidden_features),name='weight',dim=0)
        if USE_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)

        self.relu1 = nn.ReLU()

        self.hidden2 = nn.utils.weight_norm(nn.Linear(hidden_features, in_features),name='weight',dim=0)
        if USE_dropout:
            self.dropout2 = nn.Dropout(dropout_rate)
        self.relu2 = nn.ReLU()



    def forward(self, x):
        
        hidden = self.hidden1(x)
        hidden = self.dropout1(hidden)
        hidden = self.relu1(hidden)

        hidden = self.hidden2(hidden)
        hidden = hidden + x
        hidden = self.relu2(hidden)

        return hidden
        
class TRANSFORMER_CONV(torch.nn.Module):
    def __init__(self, heads, dropout):
        # Init parent
        super(TRANSFORMER_CONV, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = TransformerConv(NODE_FEATURES, EMBEDDING_SIZE, heads=heads, dropout=dropout)
        self.conv1 = TransformerConv(EMBEDDING_SIZE*heads, EMBEDDING_SIZE, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(EMBEDDING_SIZE*heads, EMBEDDING_SIZE, heads=heads, dropout=dropout)
        self.conv3 = TransformerConv(EMBEDDING_SIZE*heads, EMBEDDING_SIZE, heads=heads, dropout=dropout)
        self.conv4 = TransformerConv(EMBEDDING_SIZE*heads, EMBEDDING_SIZE, heads=heads, dropout=dropout)
        
        self.hidden1 = Linear(EMBEDDING_SIZE*heads, EMBEDDING_SIZE*2)
        self.dropout1 = nn.Dropout(0.2)
        self.relu1 = nn.ReLU()
        self.skip1 = SKIPblock(EMBEDDING_SIZE*2, EMBEDDING_SIZE*3)
        
        self.hidden2 = Linear(EMBEDDING_SIZE*2, EMBEDDING_SIZE*3)
        self.dropout2 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()
        self.skip2 = SKIPblock(EMBEDDING_SIZE*3, EMBEDDING_SIZE*2)

        self.hidden3 = Linear(EMBEDDING_SIZE*3, EMBEDDING_SIZE*4)
        self.dropout3 = nn.Dropout(0.2)
        self.relu3 = nn.ReLU()
        self.skip3 = SKIPblock(EMBEDDING_SIZE*4, EMBEDDING_SIZE*2)

        self.hidden4 = Linear(EMBEDDING_SIZE*4, EMBEDDING_SIZE*2)
        self.dropout4 = nn.Dropout(0.2)
        self.relu4 = nn.ReLU()
        self.skip4 = SKIPblock(EMBEDDING_SIZE*2, EMBEDDING_SIZE*2)

        self.forward_prediction = Linear(EMBEDDING_SIZE*2, OUTPUT_SIZE)
        self.backward_prediction = Linear(EMBEDDING_SIZE*2, OUTPUT_SIZE)
        self.gate = Linear(EMBEDDING_SIZE*2, OUTPUT_SIZE)

        self.relu_out = nn.ReLU()

    def forward(self, x, edge_index, edge_weight, total_mass, batch_index):
        
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
     
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = F.relu(hidden)

        hidden = gmp(hidden, batch_index)

        hidden = self.hidden1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.skip1(hidden)

        hidden = self.hidden2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.skip2(hidden)

        hidden = self.hidden3(hidden)
        hidden = self.dropout3(hidden)
        hidden = self.relu3(hidden)
        hidden = self.skip3(hidden)

        hidden = self.hidden4(hidden)
        hidden = self.dropout4(hidden)
        hidden = self.relu4(hidden)
        hidden = self.skip4(hidden)


        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, MASS_SHIFT)
        
        # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, MASS_SHIFT)
        
        # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = F.sigmoid(gate_hidden)

        # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)
        
        out = out.type(torch.float64)

        return out
                    
