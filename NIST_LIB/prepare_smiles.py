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

def spectrum_preparation_double(spectrum, intensity_power, output_size, operation):
    # get spectrum object and return array of specific size for prediction
    # spectrum is in shape tuple[tuple[2]]
    spectrum_output = torch.zeros(1, output_size)
    
    
    for position, intensity in spectrum:
        if position >= output_size:
            spectrum_output[0][output_size - 1] = intensity
            continue
        spectrum_output[0][int(position)] = intensity
    
    if operation == "pow":
      spectrum_output = torch.pow(spectrum_output, intensity_power)
    elif operation == "log":
      spectrum_output = spectrum_output + 1
      spectrum_output = torch.log(spectrum_output)
    else:
      spectrum_output = spectrum_output

    
    return spectrum_output.type(torch.float64)
    
def one_hot_encoding(label, num_labels):
    # make one hot encoding for one instance
    # args
    # label: int, position in one hot vector
    # num_label = int, how many groups exist
    # return: torch tensor
    tmp_zeroes = torch.zeros(num_labels)
    
    
    if type(label) is bool:
        tmp_zeroes[0] = label
        return tmp_zeroes
    if label >= num_labels:
        tmp_zeroes[num_labels - 1] = float(1)
        warnings.warn("Number of group is greater than one hot dimension representation")
        return tmp_zeroes
    elif label < 0:
      tmp_zeroes[0] = float(1)
      return tmp_zeroes
    else:
        tmp_zeroes[label] = float(1)
    return tmp_zeroes
    
    
def get_atom_features(atom):
#     result = []
    torch_result = torch.tensor([])

    atomic_number = torch.tensor([atom.GetAtomicNum()]) / 100.0
    torch_result = torch.cat((torch_result, atomic_number), 0)

    PERMITTED_LIST_OF_ATOMAS =  ['H','C','N','O','F','P', 'S', 'Unknown']
    atom_dict = {elem: index for index, elem in enumerate(PERMITTED_LIST_OF_ATOMAS)}
    
    atom_type_hot = one_hot_encoding(atom_dict.get(atom.GetSymbol(), len(atom_dict)),
                                     len(PERMITTED_LIST_OF_ATOMAS))

    torch_result = torch.cat((torch_result, atom_type_hot), 0)
    
    total_valence = atom.GetTotalValence()
    total_valence_hot = one_hot_encoding(total_valence, 8)
    # print("total_valence", total_valence)
    torch_result = torch.cat((torch_result, total_valence_hot), 0)
    
    is_aromatic_hot = one_hot_encoding(atom.GetIsAromatic(), 1)
    torch_result = torch.cat((torch_result, is_aromatic_hot), 0)
    
    
    HYBRIDIZATIONS = [Chem.HybridizationType.UNSPECIFIED,
                      Chem.HybridizationType.S, 
                      Chem.HybridizationType.SP, 
                      Chem.HybridizationType.SP2, 
                      Chem.HybridizationType.SP3, 
                      Chem.HybridizationType.SP3D, 
                      Chem.HybridizationType.SP3D2,
                      Chem.HybridizationType.OTHER]
    hybridization_dict = {elem: index for index, elem in enumerate(HYBRIDIZATIONS)}
    hybridization = atom.GetHybridization()
    hybridization_hot = one_hot_encoding(hybridization_dict.get(hybridization, len(hybridization_dict)), 8)
    torch_result = torch.cat((torch_result, hybridization_hot), 0)
    # print("hybridization", hybridization)
    
    # we adapt scale, the output of method GetFormalCharge is [-2, -1, 0, 1, 2]
    formal_charge = atom.GetFormalCharge()
    # print("foral_charge", formal_charge)
    formal_charge_hot = one_hot_encoding(formal_charge + 2, 5)
    torch_result = torch.cat((torch_result, formal_charge_hot), 0)
    
    default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    # print("default valence", default_valence)
    default_valence_hot = one_hot_encoding(default_valence, 8)
    torch_result = torch.cat((torch_result, default_valence_hot), 0)
    
    ring_size = [atom.IsInRingSize(r) for r in range(3, 8)]
    # print("ring_size", ring_size)
    ring_size_hot = torch.tensor(ring_size).type(torch.float)
    torch_result = torch.cat((torch_result, ring_size_hot), 0)
    
    attached_H = np.sum([neighbour.GetAtomicNum() == 1 for neighbour in atom.GetNeighbors()], dtype=np.uint8)
    explicit = atom.GetNumExplicitHs()
    implicit = atom.GetNumImplicitHs()
    H_num = attached_H + explicit + implicit
    # print(attached_H, explicit, implicit)
    try:
        H_hot = one_hot_encoding(H_num, 6)
    except:
        print(H_num)
        print(attached_H, explicit, implicit)
        raise Exception("Sorry, no numbers below zero") 
        

    torch_result = torch.cat((torch_result, H_hot), 0)

    return torch_result
            
    
def get_bond_features(bond, use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    
    torch_result = torch.tensor([])
    
    BOND_TYPE = [1.0, 1.5, 2.0, 3.0]
    bond_dict = {elem: index for index, elem in enumerate(BOND_TYPE)}
    bond_type_hot = one_hot_encoding(bond_dict.get(bond.GetBondTypeAsDouble(), len(bond_dict)),
                                     len(BOND_TYPE))
    torch_result = torch.cat((torch_result, bond_type_hot), 0)
    
    bond_is_conj_hot = one_hot_encoding(bond.GetIsConjugated(), 1)
#     bond_is_conj_enc = [int(bond.GetIsConjugated())]
    torch_result = torch.cat((torch_result, bond_is_conj_hot), 0)
    
    bond_is_in_ring_hot = one_hot_encoding(bond.IsInRing(), 1)
#     bond_is_in_ring_enc = [int(bond.IsInRing())]
    torch_result = torch.cat((torch_result, bond_is_in_ring_hot), 0)

    
    if use_stereochemistry == True:
        STEREO_TYPE = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        stereo_dict = {elem: index for index, elem in enumerate(STEREO_TYPE)}
        stereo_type_hot = one_hot_encoding(stereo_dict.get(str(bond.GetStereo()), len(stereo_dict)),
                                                           len(STEREO_TYPE))
        torch_result = torch.cat((torch_result, stereo_type_hot), 0)
    return torch_result
    
def create_pytorch_geometric_graph_data_list_parquet(nist_data, intensity_power, output_size, operation):
    """
    Inputs:
    
    Pandas dataframe with columns: 
    rdkit mol
    spectrum: tuple[tuple[2]]
    smiles
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for _, nist_obj in nist_data.iterrows():
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(nist_obj['smiles'])
        
        if mol == None:            
            continue
       

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        
        # the purpose is to find out one hot emb dimension
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float64)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
       
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # weight of molecul
        MW = nist_obj.get("mw", None)
        if MW == None:
            MW = Descriptors.ExactMolWt(mol)
        MW = torch.tensor(int(round(float(MW))))
        
        # construct label tensor
        y_tensor = spectrum_preparation_double(nist_obj["spect"], intensity_power, output_size, operation)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, molecular_weight = MW, y = y_tensor))

    return data_list
