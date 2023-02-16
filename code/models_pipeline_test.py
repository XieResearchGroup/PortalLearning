import numpy as np
#--------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax
#--------------------------
from torch_geometric.utils import to_dense_batch
from model_Yang_test import *
from resnet import ResnetEncoderModel
from data_tool_box import *
from model_piple_utils_test import *
#--------------------------

class DTI_pipeline(nn.Module):
    def __init__(self, all_config=None,
                 protein_descriptor=None,chem=None
               ):
        super(DTI_pipeline, self).__init__()
        # -------------------------------------------
        #         hyper-parameter
        # -------------------------------------------
        self.use_cuda = all_config['use_cuda']
        # self.contextpred_config= contextpred_config
        self.all_config = all_config
        # self.tape_related = tape_related
        # -------------------------------------------
        #         model components
        # -------------------------------------------
        #        interaction

       #          chemical decriptor
        self.ligandEmbedding = chem

        #          protein decriptor
        proteinEmbedding = protein_descriptor
        self.proteinEmbedding  = proteinEmbedding
        if all_config['protein_descriptor']=='DISAE':
            prot_embed_dim = 256
        elif all_config['protein_descriptor'] =='TAPE':
            prot_embed_dim = 768
        # else:
        #     prot_embed_dim = 1280
        if all_config['pipelinefrozen']=='transformer':
            # print(' frozen transformer in the final pipeline model')
        #     prot_embed_dim = 256
            ct = 0
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in all_config['DISAE']['frozen_list']:
                    # print('frozen module ', ct)
                    for param in m.parameters():
                        param.requires_grad = False
                else:
                    for param in m.parameters():
                        param.requires_grad = True
                # else:
        self.resnet = ResnetEncoderModel(1)
        # print('plus Resnet!')
        # self.prot_embed_transform = nn.Linear(prot_embed_dim,contextpred_config['emb_dim'])
        #        interaction
        self.attentive_interaction_pooler = AttentivePooling(300 )
        self.interaction_pooler = EmbeddingTransform(300 + prot_embed_dim, 128, 64,    0.1)
        self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)



    def forward(self, batch_protein_tokenized,batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        if self.all_config['protein_descriptor']=='DISAE':
            if self.all_config['frozen'] == 'whole':
                with torch.no_grad():
                    batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
            else:
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]

            batch_protein_repr_transformed = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(self.all_config['batch_size'],1,-1)#(batch_size,1,256)



        # ---------------ligand embedding ready -------------
        node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                   batch_chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation, batch_chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)  # (batch_size,1,300)
        # ---------------interaction embedding ready -------------
        ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(  batch_chem_graphs_repr_pooled,
                                                                                                     batch_protein_repr_transformed)  # same as input dimension

        # interaction_vector = torch.cat((prot_vec, chem_vec), dim=1)
        interaction_vector = self.interaction_pooler(
            torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        logits = self.binary_predictor(interaction_vector)  # (batch_size,2)
        return logits

