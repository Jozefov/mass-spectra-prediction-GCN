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

EMBEDDING_SIZE = 2000
NODE_FEATURES = 50
OUTPUT_SIZE = 1000
# depends on fingerprint generation
INPUT_SIZE = 2**10
INTENSITY_POWER = 0.5
MASS_SHIFT = 5 
EDGE_EMBEDDING = 10
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
    def __init__(self, in_features, hidden_features, bottleneck_factor=0.5, USE_dropout=True, dropout_rate = 0.2):
        super().__init__()
        #only need to change shape of the residual if num_channels changes (i.e. in_c != out_c)
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]
        
        self.batchNorm1 = nn.BatchNorm1d(in_features)
        self.relu1 = nn.ReLU()
        if USE_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden1= nn.utils.weight_norm(nn.Linear(in_features, int(hidden_features * bottleneck_factor)),name='weight',dim=0)
        
        self.batchNorm2 = nn.BatchNorm1d(int(hidden_features * bottleneck_factor))
        self.relu2 = nn.ReLU()
        if USE_dropout:
            self.dropout2 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.utils.weight_norm(nn.Linear(int(hidden_features * bottleneck_factor), in_features),name='weight',dim=0)



    def forward(self, x):
        
        hidden = self.batchNorm1(x)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.hidden1(hidden)

        hidden = self.batchNorm2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.hidden2(hidden)

        hidden = hidden + x

        return hidden
        
class DENSE_BIG(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(DENSE_BIG, self).__init__()
        torch.manual_seed(42)

        
        self.hidden1 = nn.Linear(INPUT_SIZE, EMBEDDING_SIZE)
        self.skip1 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip2 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip3 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip4 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip5 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip6 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip7 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.backward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.gate = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)

        self.relu_out = nn.ReLU()
        

    def forward(self, x, total_mass):

        hidden = self.hidden1(x)
        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)
        
        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, MASS_SHIFT)
        
        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, MASS_SHIFT)
        
        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = F.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)
        
        out = out.type(torch.float64)

        return out
        
class CONV_BIG(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(CONV_BIG, self).__init__()
        torch.manual_seed(42)

        EMBEDDING_SIZE_REDUCED = int(EMBEDDING_SIZE*0.15)

        # GCN layers
        self.initial_conv = GCNConv(NODE_FEATURES, EMBEDDING_SIZE_REDUCED)
        self.conv1 = GCNConv(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE_REDUCED)
        self.reluconv1 = nn.ReLU()
        self.conv2 = GCNConv(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE_REDUCED)
        self.reluconv2 = nn.ReLU()
        self.conv3 = GCNConv(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE_REDUCED)
        self.reluconv3 = nn.ReLU()
        self.conv4 = GCNConv(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE_REDUCED)
        self.reluconv4 = nn.ReLU()
    
        self.bottleneck = Linear(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE)

        self.skip1 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip2 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip3 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip4 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip5 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip6 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip7 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.backward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.gate = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)

        self.relu_out = nn.ReLU()

    def forward(self, x, edge_index, edge_weight, total_mass, batch_index):
        
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = self.reluconv1(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.reluconv2(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.reluconv3(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.reluconv4(hidden)
     
        
        hidden = gap(hidden, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)
        
        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, MASS_SHIFT)
        
        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, MASS_SHIFT)
        
        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = F.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)
        
        out = out.type(torch.float64)
        return out

class SKIPGAT(nn.Module):
    def __init__(self, in_features, hidden_features, heads, USE_dropout=True, dropout_rate = 0.1):
        super().__init__()
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]

        self.relu1 = nn.ReLU()
        if USE_dropout:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads, dropout=dropout_rate)
        else:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads)
        
        self.relu2 = nn.ReLU()
        if USE_dropout:
            self.conv2 = GATConv(hidden_features*heads, int(in_features/heads), heads=heads, dropout=dropout_rate)
        else:
            self.conv2 = GATConv(hidden_features*heads, int(in_features/heads), heads=heads)

    def forward(self, x, edge_index, edge_weight):
        
        hidden = self.relu1(x)
        hidden = self.conv1(hidden, edge_index, edge_weight)
        
        hidden = self.relu2(hidden)
        hidden = self.conv2(hidden, edge_index, edge_weight)
        hidden = hidden + x
        
        return hidden
        
class GAT_DEEP_BIG(torch.nn.Module):
    def __init__(self, heads):
        # Init parent
        super(GAT_DEEP_BIG, self).__init__()
        torch.manual_seed(42)
        
        EMBEDDING_SIZE_REDUCED = int(EMBEDDING_SIZE*0.15)
        # GCN layers
        self.initial_conv = GATConv(NODE_FEATURES, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat1 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat2 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat3 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat4 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.mean_conv = GATConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads, concat=False)
        self.mean_relu = nn.ReLU()

        self.bottleneck = Linear(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE)

        self.skip1 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip2 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip3 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip4 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip5 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip6 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip7 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.backward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.gate = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)

        self.relu_out = nn.ReLU()


    def forward(self, x, edge_index, edge_weight, total_mass, batch_index):
        
        hidden = self.initial_conv(x, edge_index, edge_weight)
     
        # Other Conv layers
        hidden = self.skipgat1(hidden, edge_index, edge_weight)
        hidden = self.skipgat2(hidden, edge_index, edge_weight)
        hidden = self.skipgat3(hidden, edge_index, edge_weight)
        hidden = self.skipgat4(hidden, edge_index, edge_weight)
        hidden = self.mean_conv(hidden, edge_index, edge_weight)
        

        hidden = gmp(hidden, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)
        
        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, MASS_SHIFT)
        
        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, MASS_SHIFT)
        
        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = F.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)
        
        out = out.type(torch.float64)
        return out
        
        
class TRANSFORMER_CONV_MESSAGE_BIG(torch.nn.Module):
    def __init__(self, heads, dropout):
        # Init parent
        super(TRANSFORMER_CONV_MESSAGE_BIG, self).__init__()
        torch.manual_seed(42)
        EMBEDDING_SIZE_REDUCED = int(EMBEDDING_SIZE*0.1)

        # GCN layers
        self.initial_conv = TransformerConv(NODE_FEATURES, EMBEDDING_SIZE_REDUCED, heads=heads, beta=True, dropout=dropout, edge_dim=EDGE_EMBEDDING)
        self.conv1 = TransformerConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads, beta=True, dropout=dropout, edge_dim=EDGE_EMBEDDING)
        self.conv2 = TransformerConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads, beta=True, dropout=dropout, edge_dim=EDGE_EMBEDDING)
        self.conv3 = TransformerConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads, beta=True, dropout=dropout, edge_dim=EDGE_EMBEDDING)
        self.conv4 = TransformerConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, concat=False, heads=heads, beta=True, dropout=dropout, edge_dim=EDGE_EMBEDDING)
        
        self.bottleneck = Linear(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE)

        self.skip1 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip2 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip3 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip4 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip5 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip6 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip7 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.backward_prediction = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.gate = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)

        self.relu_out = nn.ReLU()

    def forward(self, x, edge_index, edge_weight, total_mass, batch_index):
        
        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.relu(hidden)
     
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index, edge_weight)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index, edge_weight)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index, edge_weight)
        hidden = F.relu(hidden)
        hidden = self.conv4(hidden, edge_index, edge_weight)
        
      
        hidden = gap(hidden, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)
        
        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, MASS_SHIFT)
        
        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, MASS_SHIFT)
        
        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = F.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)
        
        out = out.type(torch.float64)
        return out
        
class GLUblock_1D(nn.Module):
    def __init__(self, k, in_c, out_c):
        super().__init__()

        if in_c == out_c:
            self.use_proj=0
        else:
            self.use_proj=1
        self.convresid=nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=1),name='weight',dim=0)
        
        self.leftpad = nn.ConstantPad1d((k-1, 0),0.0)
      
        self.convx1a = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k),name='weight',dim=0)
        self.convx2a = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k),name='weight',dim=0)


    def forward(self, x):
        residual = x
        
        if self.use_proj==1:
            residual=self.convresid(residual)
        x=self.leftpad(x) 
        
        x1 = self.convx1a(x) 
        x2 = self.convx2a(x)
        x2 = torch.sigmoid(x2)
        
        x=torch.mul(x1,x2)
        return x+residual
        
class GATTED_GAT_BIG(torch.nn.Module):
    def __init__(self, kernel_size, number_of_gates, number_of_channels, heads):
        super(GATTED_GAT_BIG, self).__init__()
        # Init parent
        torch.manual_seed(42)

        EMBEDDING_SIZE_REDUCED = int(EMBEDDING_SIZE*0.15)
        # GCN layers
        self.initial_conv = GATConv(NODE_FEATURES, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat1 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat2 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat3 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.skipgat4 = SKIPGAT(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads)
        self.mean_conv = GATConv(EMBEDDING_SIZE_REDUCED*heads, EMBEDDING_SIZE_REDUCED, heads=heads, concat=False)
        self.mean_relu = nn.ReLU()
    
        self.bottleneck = Linear(EMBEDDING_SIZE_REDUCED, EMBEDDING_SIZE)

        self.skip1 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip2 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip3 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip4 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip5 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.skip6 = SKIPblock(EMBEDDING_SIZE, EMBEDDING_SIZE)

        self.relu_out_resnet = nn.ReLU()
        self.bottleneck2 = Linear(EMBEDDING_SIZE, OUTPUT_SIZE)
        self.relu_bottlenec2 = nn.ReLU()

        self.GLUlayers=self.make_GLU_layers(kernel_size, number_of_channels, number_of_gates)
        self.out = nn.ReLU()

    def make_GLU_layers(self, kernel_size, num_channels, num_layers):
        layers = [GLUblock_1D(kernel_size, num_channels, num_channels) for i in range(num_layers)]
        return nn.Sequential(*layers)

        self.relu_out = nn.ReLU()

    def forward(self, x, edge_index, edge_weight, total_mass, batch_index):
        
        hidden = self.initial_conv(x, edge_index, edge_weight)
     
        # Other Conv layers
        hidden = self.skipgat1(hidden, edge_index, edge_weight)
        hidden = self.skipgat2(hidden, edge_index, edge_weight)
        hidden = self.skipgat3(hidden, edge_index, edge_weight)
        hidden = self.skipgat4(hidden, edge_index, edge_weight)
        hidden = self.mean_conv(hidden, edge_index, edge_weight)
        
        hidden = gap(hidden, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        
        hidden = self.relu_out_resnet(hidden)
        hidden = self.bottleneck2(hidden)
        hidden = self.relu_bottlenec2(hidden)
     
        hidden = torch.unsqueeze(hidden, 1)
        # Gatted layers
        
        hidden = self.GLUlayers(hidden)
        
        hidden = torch.squeeze(hidden, 1)
       
        hidden = mask_prediction_by_mass(total_mass, hidden, MASS_SHIFT)
        
        hidden = self.relu_out_resnet(hidden)
        hidden = hidden.type(torch.float64)

        return hidden
