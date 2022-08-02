import argparse
import random
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
# --------------
from utils import *
from data_tool_box import *
from utils_MAML import  *
# --------------
import torch
from torch import  nn
from torch.nn import functional as F
from rdkit import Chem
# -------------
from models_MAML_test import  *
from models_pipeline import *
from model_piple_utils import *
from model_Yang_test import *


# -------------------------------
class portal_four_universe_Ensemble(nn.Module):
    def __init__(self,args,protein_descriptor):
        super(portal_four_universe_Ensemble, self).__init__()
        self.args=args
        # .......... LOAD DISAE ..........
        path = '../data/illuminating/'
        chem_descriptor = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')
        self.model1 = DTI_model_MAML(all_config=args, model=protein_descriptor, chem=chem_descriptor)
        self.model2 = DTI_model_MAML(all_config=args, model=protein_descriptor, chem=chem_descriptor)
        self.model3 = DTI_model_MAML(all_config=args, model=protein_descriptor, chem=chem_descriptor)
        self.model1.load_state_dict(torch.load(path+'model_chosen/model1.dat'))
        self.model2.load_state_dict(torch.load(path + 'model_chosen/model2.dat'))
        self.model3.load_state_dict(torch.load(path + 'model_chosen/model3.dat'))
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,batch_data):
        batch_chem_graphs_spt, batch_protein_tokenized_spt = get_repr_DTI_illum(batch_data,
                                                                              self.args)
        batch_logits1 = self.model1(batch_protein_tokenized_spt, batch_chem_graphs_spt)
        batch_logits2 = self.model2(batch_protein_tokenized_spt, batch_chem_graphs_spt)
        batch_logits3 = self.model3(batch_protein_tokenized_spt, batch_chem_graphs_spt)
        batch_logits = batch_logits1+ batch_logits2+batch_logits3
        return batch_logits1.detach().cpu(), batch_logits2.detach().cpu(),batch_logits3.detach().cpu()
        # return batch_logits3.detach().cpu()
        # return self.softmax(batch_logits.detach().cpu()).numpy()[:,1]
class portal_four_universe_single(nn.Module):
    def __init__(self,args,protein_descriptor):
        super(portal_four_universe_single, self).__init__()
        self.args=args
        # .......... LOAD DISAE ..........
        path = '../data/illuminating/'
        chem_descriptor = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')
        self.model3 = DTI_model_MAML(all_config=args, model=protein_descriptor, chem=chem_descriptor)
        self.model3.load_state_dict(torch.load(path + 'model_chosen/model3.dat'))
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,batch_data):
        batch_chem_graphs_spt, batch_protein_tokenized_spt = get_repr_DTI_illum(batch_data,
                                                                              self.args)
        batch_logits3 = self.model3(batch_protein_tokenized_spt, batch_chem_graphs_spt)

        return batch_logits3.detach().cpu()
        # return self.softmax(batch_logits.detach().cpu()).numpy()[:,1]

def get_repr_DTI_illum(batch_data,args):
    #  . . . .  chemicals  . . . .
    chem_graph_list = batch_data['drug-list']
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=len(chem_graph_list),
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    #  . . . .  proteins  . . . .
    protein_tokenized = torch.tensor(batch_data['protein']).repeat(len(chem_graph_list),1)
    #  . . . .  to return  . . . .
    chem_graphs = chem_graphs.to(args['device'])
    protein_tokenized = protein_tokenized.to(args['device'])
    return chem_graphs, protein_tokenized
