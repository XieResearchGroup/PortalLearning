import numpy as np
import pandas as pd
#--------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax
from torch.nn.utils.rnn import pad_sequence
#--------------------------
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
#--------------------------
from model_Yang import *
from ligand_graph_features import *

class EmbeddingTransform2(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.1):
        super(EmbeddingTransform2, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size)
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden
class EmbeddingTransform(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.1):
        super(EmbeddingTransform, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size)
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden

class AttentivePooling2(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self,embedding_length = 300):
        super(AttentivePooling2, self).__init__()
        self.embedding_length = embedding_length
        self.U = nn.Parameter(torch.zeros(self.embedding_length, self.embedding_length))

    def forward(self, protein, ligand):
        """ Calculate attentive pooling attention weighted representation and

        """

        U= self.U.expand(protein.size(0), self.embedding_length,self.embedding_length)
        Q = protein
        A = ligand
        G = torch.tanh(torch.bmm(torch.bmm(Q, U), A.transpose(1,2)))
        g_q = G.max(axis=2).values
        g_a = G.max(axis=1).values

        def get_attention_score(g_q,Q):
            g_q_masked = g_q.masked_fill(g_q == 0, -1e9)
            sigma_q = F.softmax(g_q_masked)
            prot_repr = Q * sigma_q[:, :, None]


            return sigma_q, prot_repr

        sigma_q, prot_vec = get_attention_score(g_q,Q)
        sigma_a, chem_vec = get_attention_score(g_a,A)

        return sigma_q, prot_vec, sigma_a, chem_vec
class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, chem_hidden_size=128,prot_hidden_size=256):
        super(AttentivePooling, self).__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))

    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.

        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)

        Returns:
            (rep_1, attn_1): attention weighted representations and attention scores
            for the first input
            (rep_2, attn_2): attention weighted representations and attention scores
            for the second input
        """
        # logging.debug("AttentivePooling first {0}, second {1}".format(first.size(), second.size()))
        param = self.param.expand(first.size(0), self.chem_hidden_size,self.prot_hidden_size)
        # logging.debug("AttentivePooling params: {0}".format(param.size()))
        wm1 = torch.tanh(torch.bmm(second,param.transpose(1,2)))
        wm2 = torch.tanh(torch.bmm(first,param))
        # logging.debug("Wm1 {}, Wm2 {} before softmax".format(wm1.size(),wm2.size()))
        score_m1 = F.softmax(wm1,dim=2)
        score_m2 = F.softmax(wm2,dim=2)
        # logging.debug("score_m1 {}, score_m2 {}".format(score_m1.size(),score_m2.size()))
        rep_first = first*score_m1
        rep_second = second*score_m2
        # logging.debug("AttentivePooling reps: {0}, {1}".format(rep_first.size(), rep_second.size()))

        return ((rep_first, score_m1), (rep_second, score_m2))

class distMtx_core_module(nn.Module):
    def __init__(self,dim =20,num_class=3,
                 feat_mode='attentive-pool'):
        super(distMtx_core_module, self).__init__()
        self.feat_mode=feat_mode
        self.attpool = AttentivePooling2(embedding_length=dim)
        self.ffn= torch.nn.Linear(dim,num_class)

    def create_pairwise_embed(self,embed_a, embed_b, feat_mode=None):
        if feat_mode == 'multiply':
            feat = embed_a[:, :, None, :] * embed_b[:, None, :, :]
        if feat_mode == 'attentive-pool':
            sigma_a, a_vec, sigma_b, b_vec = self.attpool(embed_a, embed_b)
            feat = a_vec[:, :, None, :] * b_vec[:, None, :, :]
        return feat

    def forward(self, embed_a, embed_b):
        pairwise_feat = self.create_pairwise_embed(embed_a, embed_b, self.feat_mode)
        logits = self.ffn(pairwise_feat)
        return logits



class DTI_distMtx_classifier(nn.Module):
    def __init__(self,dim =20,
                 feat_mode='attentive-pool',pred_mode='binary',protein_descriptor=None,chem_descriptor = None,
                 frozen='whole',cwd=None,chem_pretrained='nope'):
        super(DTI_distMtx_classifier, self).__init__()
        if pred_mode == 'binary':
            num_class = 2
        else:
            num_class = 9 # 9 for even prob 0.1 setting,34 for PDNET
        self.pred_DistMtx = distMtx_core_module(dim = 300,num_class=num_class,feat_mode=feat_mode)
        self.protein_descriptor = protein_descriptor
        # self.chem_decriptor = GNN(num_layer=5,emb_dim=300,JK='last',drop_ratio=0.5,gnn_type='gin')
        self.chem_decriptor = chem_descriptor

        self.prot_transform = torch.nn.Linear(dim,300) # transform to same dim as contextpred
        self.frozen=frozen
        if frozen =='encoder-whole':
            # print('frozen TAPE encoder whole')
            for n, m in self.protein_descriptor.named_children():
                if n == 'encoder':
                    for param in m.parameters():
                        param.requires_grad = False


    def forward(self,batch_input,device):
        # ---------------protein embedding -------------------
        if self.frozen=='whole':
            with torch.no_grad():
                embed_full_prot = self.protein_descriptor(batch_input['tokenized-padded'])[0][:,1:-1,:]# with beining and end tokens
        else:
            # print('updating..')
            embed_full_prot = self.protein_descriptor(batch_input['tokenized-padded'])[0][:, 1:-1,:]  # with beining and end tokens
        embed_bs_prot = torch.bmm(batch_input['binding site selection matrix|prot'],embed_full_prot)
        embed_bs_prot_T = self.prot_transform(embed_bs_prot)
        # ---------------chemical embedding -------------------
        chem_graphs_in = batch_input['chem graph loader']
        embed_full_chem = self.chem_decriptor(chem_graphs_in.x, chem_graphs_in.edge_index,
                              chem_graphs_in.edge_attr
                              )
        graph_repr_masked, mask_graph = to_dense_batch(embed_full_chem, chem_graphs_in.batch)
        embed_bs_chem = torch.bmm(batch_input['binding site selection matrix|chem'], graph_repr_masked)
        logits = self.pred_DistMtx(embed_bs_prot_T,embed_bs_chem)
        return  logits

class DTI_distMtx_classifier0(nn.Module):
    def __init__(self,dim =20,
                 feat_mode='attentive-pool',pred_mode='binary',protein_descriptor=None,chem_descriptor = None,
                 frozen='whole',cwd=None,chem_pretrained='nope'):
        super(DTI_distMtx_classifier0, self).__init__()
        if pred_mode == 'binary':
            num_class = 2
        else:
            num_class = 9 # 9 for even prob 0.1 setting,34 for PDNET
        self.pred_DistMtx = distMtx_core_module(dim = 300,num_class=num_class,feat_mode=feat_mode)
        self.protein_descriptor = protein_descriptor
        self.chem_decriptor = GNN(num_layer=5,emb_dim=300,JK='last',drop_ratio=0.5,gnn_type='gin')
        # self.chem_decriptor = chem_descriptor

        self.prot_transform = torch.nn.Linear(dim,300) # transform to same dim as contextpred
        self.frozen=frozen
        if frozen =='encoder-whole':
            # print('frozen TAPE encoder whole')
            for n, m in self.protein_descriptor.named_children():
                if n == 'encoder':
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self,batch_input,device):
        # ---------------protein embedding -------------------
        if self.frozen=='whole':
            with torch.no_grad():
                embed_full_prot = self.protein_descriptor(batch_input['tokenized-padded'])[0][:,1:-1,:]# with beining and end tokens
        else:
            # print('updating..')
            embed_full_prot = self.protein_descriptor(batch_input['tokenized-padded'])[0][:, 1:-1,:]  # with beining and end tokens
        embed_bs_prot = torch.bmm(batch_input['binding site selection matrix|prot'],embed_full_prot)
        embed_bs_prot_T = self.prot_transform(embed_bs_prot)
        # ---------------chemical embedding -------------------
        chem_graphs_in = batch_input['chem graph loader']
        embed_full_chem = self.chem_decriptor(chem_graphs_in.x, chem_graphs_in.edge_index,
                              chem_graphs_in.edge_attr
                              )
        graph_repr_masked, mask_graph = to_dense_batch(embed_full_chem, chem_graphs_in.batch)
        embed_bs_chem = torch.bmm(batch_input['binding site selection matrix|chem'], graph_repr_masked)
        logits = self.pred_DistMtx(embed_bs_prot_T,embed_bs_chem)
        return  logits