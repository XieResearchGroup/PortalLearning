import argparse
from utils import *
from data_tool_box import *
from utils_MAML import  *
# --------------
import torch
from torch import  nn
from torch.nn import functional as F
from rdkit import Chem
# -------------
from models_MAML import  *
from models_pipeline import *
from model_piple_utils import *
from model_Yang import *
from utils_esemble import *

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from scipy.stats import spearmanr

parser = argparse.ArgumentParser("Test on DRD3")
parser.add_argument('--cwd', type=str,  default='')
opt = parser.parse_args()

def get_result(truescore,dockscore):
    fpr, tpr, thresholds = roc_curve(truescore, dockscore, pos_label=1)
    prec, recall, thresholds2 = precision_recall_curve(truescore, dockscore, pos_label=1)
    alphafold = [fpr, tpr, 'a',prec,recall]
    return alphafold

def scores(portal):
    roc = round(metrics.auc(portal[0],portal[1]),2)
    pr = round(metrics.auc(portal[4],portal[3]),2)

    return roc, pr

DRD_path =  '../data/DRD/'
input_file = DRD_path +'drd_map.pkl'
batchsize= 5
threshold=0.58

DTI_classifier_config ={
    'chem_pretrained':'nope',
    'protein_descriptor':'DISAE',
    'DISAE':{'frozen_list':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},
    'frozen':'none',
    'phase1ResiDist':'none',
    'phase2DTIDist':'none'
}

args={'cwd':opt.cwd,'input_file':'','batchsize':batchsize,'use_cuda':True,
    'chem_pretrained':'nope',
    'protein_descriptor':'DISAE',
    'DISAE':{'frozen_list':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},
    'frozen':'none',
    'phase1ResiDist':'none',
    'phase2DTIDist':'none',
     'device':'cuda'}

proteins= load_pkl(input_file)
proteins = { 'P35462 (DRD3_HUMAN)':proteins['P35462 (DRD3_HUMAN)'] }
dark_proteins = list(proteins.keys())

drug_dict = pd.Series(load_pkl(DRD_path +'drug_dict.pkl'))
ikey_list = drug_dict.index.values.tolist()
smiles_list = drug_dict.values.tolist()
graph_list = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph_data_obj_simple(mol)
    graph_list.append(graph)


softmax = torch.nn.Softmax(dim=1)
protein_descriptor, prot_tokenizer = load_DISAE('')
portal = portal_four_universe_single(args,protein_descriptor)
portal = portal.to(args['device'])
bound = int(len(ikey_list) / args['batchsize'])



wetlab = pd.read_csv('../data/DRD/DRD3_affinity_data-wetlab.csv')
threshold = 1000
wetlab['label'] = wetlab.iloc[:,-1]<threshold
wetlab['label'] = wetlab.label.astype('int')
true_score = wetlab['label'].values
print('Predicting DRD3 with trained weights........')
screened_pfam = {}

for protein in dark_proteins:

    protein_tokenized = prot_tokenizer.encode(proteins[protein])
    one_protein=pd.DataFrame()
    for j in range(bound):

        batch_drug_name=pd.DataFrame()
        batch_drug_graph = graph_list[j*args['batchsize']:(j+1)*args['batchsize']]
        data={'drug-list':batch_drug_graph,'protein':protein_tokenized}
        log3 = portal(data)

        batch_drug_name['score3']= softmax(log3)[:,1].numpy()

        batch_drug_name.index = ikey_list[j*args['batchsize']:(j+1)*args['batchsize']]
        one_protein= pd.concat([one_protein,batch_drug_name])
    screened_pfam[protein]= one_protein

df_all = pd.DataFrame()
for protein in dark_proteins:
    df1 = screened_pfam[protein]
     # df1['binding-3']= df1['score3']>threshold
    df1['protein']=protein
    df_all = df_all.append(df1)


portal_df=df_all
portal_result = get_result(true_score, portal_df['score3'])
roc, pr = scores(portal_result) # 3

ki = wetlab[wetlab['D3R (ki in nM)']!=99999]['D3R (ki in nM)'].values
ki_label = wetlab[wetlab['D3R (ki in nM)']!=99999]['label'].values

portal_df= portal_df.reset_index(drop=True)
portal_score = portal_df[wetlab['D3R (ki in nM)']!=99999]['score3'].values
spear = round(spearmanr(ki, portal_score)[0],2)

print(f'ROC-AUC {roc}, PR-AUC {pr}, SpearmanCorrelation{spear}')