import pandas as pd
import numpy as np
# import pickle
import pickle5 as pickle
import json
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch
import torch
from typing import Sequence, Tuple, List, Union

from ligand_graph_features import *

##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data

##### pickle modules #####
def save_dict_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f)
  return  dict

##### for distance prediction #####
def read_binding_site_distance_mtx(file):
    i = 0
    content = []
    with open(file, 'r') as f:
        for line in f:

            if i == 1:
                columns = line.split("\t")
            if i == 2:
                seq = line.split('\t')[1].split('\n')[0]
            if i > 2:
                content.append(line.split("\t")[:-1])
            i = i + 1
    columns = columns[:-2] + ['x1', 'y1', 'z1'] + ['x2', 'y2', 'z2']
    columns[2] = 'receptor_aa_value'
    columns[3] = 'receptor_aa_atom_value'
    columns[4] = 'receptor_bb_index'
    columns[5] = 'receptor_bb_value'
    columns[6] = 'receptor_bb_atom_value'
    mtx = pd.DataFrame(content)
    mtx.columns = columns
    number = [1, 4, 7, 8, 9, 10, 11, 12]
    for i in number:
        mtx[columns[i]] = pd.to_numeric(mtx[columns[i]])

    return mtx, seq

def generate_batch_input_DTIdist(batch_data,batch_size,tokenizer,pred_mode,protein_descriptor,chem_pdb_file_path,device):
    #     ---------binding site selection matrix-protein---------------
    sizes_prot = []
    for i in range(batch_size):
        sizes_prot.append(batch_data[i]['selection matrix'][protein_descriptor].shape)
    max_b_pos_prot, max_seq_len_prot = list(np.max(np.array(sizes_prot), axis=0))
    #     ---------binding site selection matrix---------chem--------
    sizes_chem = []
    for i in range(batch_size):
        sizes_chem.append(batch_data[i]['selection matrix']['chem'].shape)
    max_b_pos_chem, max_seq_len_chem = list(np.max(np.array(sizes_chem), axis=0))
    #  -------set up ----------------
    protein_tokenized_unpad = []
    select_mtx_prot = torch.zeros(batch_size, max_b_pos_prot, max_seq_len_prot)
    targets = torch.zeros(batch_size, max_b_pos_prot, max_b_pos_chem)
    unpad_targets = []
    masks_targets = torch.zeros_like(targets)
    chem_graph_list = []
    sizes = []
    for i in range(batch_size):
        # i = 1

        #     ---------binding site selection matrix-protein---------------
        select_prot = torch.tensor(batch_data[i]['selection matrix'][protein_descriptor].toarray())
        select_mtx_prot[i, :select_prot.shape[0], :select_prot.shape[1]] = select_prot
        #     ---------binding site selection matrix---------chem--------

        select_mtx_chem = torch.zeros(batch_size, max_b_pos_chem, max_seq_len_chem)
        select_chem = torch.tensor(batch_data[i]['selection matrix']['chem'].toarray())
        select_mtx_chem[i, :select_chem.shape[0], :select_chem.shape[1]] = select_chem
        #     -----------chem graphs  ---------------
        pdb_file = batch_data[i]['chem pdb file']
        mol = Chem.MolFromPDBFile(chem_pdb_file_path + pdb_file, removeHs=False)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
        #     ---------tokenized sequence-----------------
        protein_tokenized_unpad.append(torch.tensor(tokenizer.encode(batch_data[i]['seq'][protein_descriptor])))
        #     ---------taget and target mask-----------------
        t = torch.tensor(batch_data[i]['label'][pred_mode].toarray())
        m = torch.ones_like(t)
        targets[i, :t.shape[0], :t.shape[1]] = t
        unpad_targets.append(t)
        masks_targets[i, :m.shape[0], :m.shape[1]] = m
        sizes.append(t.shape)
    #     ---------tokenized sequence in batch-----------------
    protein_tokenized_pad = pad_sequence(protein_tokenized_unpad, padding_value=0)
    protein_tokenized = protein_tokenized_pad.T
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_size,
                                    shuffle=False)
    #  ----------- chem graph in batch -----------
    for batch in chem_graphs_loader:
        chem_graphs = batch
    batch_input = {'tokenized-padded': protein_tokenized.to(device),
                   'chem graph loader': chem_graphs.to(device),
                   'binding site selection matrix|prot': select_mtx_prot.to(device),
                   'binding site selection matrix|chem': select_mtx_chem.to(device),
                   'targets-padded': targets.long().to(device),
                   'targets-mask': masks_targets.to(device)}
    return batch_input,unpad_targets,sizes

##### DTI #####
def read_binding_site_distance_mtx_DTI(file):
    i=0
    content = []
    with open(file, 'r') as f:
     try:
        for line in f:
            if i==1:
                columns = line.split("\t")
            if i ==2:
                seq = line.split('\t')[1].split('\n')[0]
            if i >2 :
                content.append(line.split("\t")[:-1])
            i = i+1
     except:
        content=[]
    columns = columns[:-2] +['x1','y1','z1'] +['x2','y2','z2']
    columns[6]='receptor_bb_index'

    if len(content)>0:
        mtx = pd.DataFrame(content)
        mtx.columns = columns
        number = [1,6,7,8,9,10,11,12]
        for i in number:
            mtx[columns[i]] = pd.to_numeric(mtx[columns[i]])
        return mtx, seq
    else:
        return '',''
#------------------
#  read data
#------------------

def load_training_data(exp_path,debug_ratio):
    def load_data(exp_path,file,debug_ratio):
        dataset = pd.read_csv(exp_path +file)
        cut = int(dataset.shape[0] * debug_ratio)
        print(file[:-3] + ' size:', cut)
        return dataset.iloc[:cut,:]

    train = load_data(exp_path,'train.csv',debug_ratio)
    dev   = load_data(exp_path,'dev.csv',debug_ratio)
    test  = load_data(exp_path,'test.csv',debug_ratio)

    return train, dev, test

def get_repr_DTI(batch_data,tokenizer,chem_dict,protein_dict,prot_descriptor_choice):
    #  . . . .  chemicals  . . . .
    chem_smiles = chem_dict[batch_data['InChIKey'].values.tolist()].values.tolist()
    chem_graph_list = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    #  . . . .  proteins  . . . .
    if prot_descriptor_choice =='DISAE':
        uniprot_list = batch_data['uniprot+pfam'].values.tolist()
        protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])

    elif prot_descriptor_choice == 'TAPE':
        uniprot_list = batch_data['uniprot+pfam'].values.tolist()
        protein_tokenized_unpad = [torch.tensor(tokenizer.encode(protein_dict[uni])[1:-1]) for uni in uniprot_list]
        protein_tokenized_pad = pad_sequence(protein_tokenized_unpad, padding_value=0)
        # protein_tokenized = protein_tokenized_pad[:, :-1]
        protein_tokenized = protein_tokenized_pad.T
    else:
        batch_seq = list(zip(list(protein_dict[batch_data['uniprot+pfam']].index),
                             protein_dict[batch_data['uniprot+pfam']].values.tolist()))
        batch_labels, batch_strs, protein_tokenized = tokenizer(batch_seq)
    return chem_graphs, protein_tokenized


def mask_esm_before_position_wise_attention(batch_protein_repr_compressed,batch_protein_tokenized,emb_dim,use_cuda):

    #mask padded position
    ones = torch.ones([batch_protein_repr_compressed.shape[0],batch_protein_tokenized.shape[1]-2])

    if use_cuda and torch.cuda.is_available():
        ones= ones.to('cuda')
    mask = ones.masked_fill(batch_protein_tokenized[:,1:-1] == 1, 0)
    mask_full = mask.unsqueeze(2).repeat(1, 1, emb_dim)
    return torch.mul(batch_protein_repr_compressed,mask_full)

def mask_tape_before_position_wise_attention(batch_protein_repr_compressed,batch_protein_tokenized,emb_dim,use_cuda):

    #mask padded position
    ones = torch.ones([batch_protein_repr_compressed.shape[0],batch_protein_tokenized.shape[1]])

    if use_cuda and torch.cuda.is_available():
        ones= ones.to('cuda')
    mask = ones.masked_fill(batch_protein_tokenized==0, 0)
    mask_full = mask.unsqueeze(2).repeat(1, 1, emb_dim)
    return torch.mul(batch_protein_repr_compressed,mask_full)
