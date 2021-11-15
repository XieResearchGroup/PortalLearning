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
from models_MAML import  *
from models_pipeline import *
from model_piple_utils import *
from model_Yang import *
# -------------------------------
parser = argparse.ArgumentParser("undruggable application")
# ... # ...admin # ...# ...# ...
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--cwd', type=str,  default='D:/Projects_on_going/distance-matrix/',
                    help='define your own current working directory,i.e. where you put your scirpt')
parser.add_argument('--inputfile',type=str, default = 'undruggableunipfam2triplet.pkl')
parser.add_argument('--step',type=int,default= 3000)
parser.add_argument('--start',type=int,default= 0)
parser.add_argument('--lead_num',type=int,default= 10)
parser.add_argument('--lead_threshold',type=float,default = 0.58)
parser.add_argument('--batchsize',type=int,default=22)
parser.add_argument('--model_no',type=int,default=2)
opt = parser.parse_args()
# -------------------------------
args = {}
args.update(vars(opt))
DTI_classifier_config ={
    'chem_pretrained':'nope',
    'protein_descriptor':'DISAE',
    'DISAE':{'frozen_list':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},
    'frozen':'none',
    'phase1ResiDist':'none',
    'phase2DTIDist':'none'
}
args.update(DTI_classifier_config)
seed = 705
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    args['device']='cuda'
else:
    args['device'] = 'cpu'
checkpoint_path = set_up_exp_folder(opt.cwd,exp ='undruggable_app/')
config_file = checkpoint_path + 'config.json'
save_json(vars(opt),    config_file)
# -------------------------------
class portal_four_universe(nn.Module):
    def __init__(self,args,protein_descriptor):
        super(portal_four_universe, self).__init__()
        self.args=args
        # .......... LOAD DISAE ..........
        path = args['cwd'] +'data/illuminating/'
        chem_descriptor = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')
        self.model = DTI_model_MAML(all_config=args, model=protein_descriptor, chem=chem_descriptor).to( args['device'])
        # self.model.load_state_dict(torch.load(path+'maml_model.dat'))
        self.model.load_state_dict(torch.load(path+'model_chosen/model'+str(args['model_no'])+'.dat'))
        print('------------------ loaded model: ',str(args['model_no']) )
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,batch_data):
        batch_chem_graphs_spt, batch_protein_tokenized_spt = get_repr_DTI_illum(batch_data,
                                                                              self.args)
        batch_logits = self.model(batch_protein_tokenized_spt, batch_chem_graphs_spt)
        return batch_logits.detach().cpu().numpy().astype(np.float32)
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

if __name__ == '__main__':
    illum_path = args['cwd'] + 'data/illuminating/'
    undruggable = load_pkl(illum_path+'undruggable/'+args['inputfile'])
    proteins =list( undruggable.keys())[args['start']:]
    protein_descriptor, prot_tokenizer = load_DISAE(args['cwd'])
    # -------------------------------------------------------------------------------------------------
    drug_dict = pd.Series(load_pkl(illum_path + 'drug_dict.pkl'))
    ikey_list = drug_dict.index.values.tolist()
    smiles_list = drug_dict.values.tolist()
    graph_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        graph_list.append(graph)
# --------------------------------- screening
    bound = int(len(ikey_list) / args['batchsize'])
    portal = portal_four_universe(args,protein_descriptor)
    i=args['start']
    screened = pd.DataFrame()
    for protein in proteins:
        print('----------------------------- protein: ',i,'             ',protein)
        # stime = time.time()
        # screened_pfam = {}
        # dark_proteins = dark_space[dark_space['pfam']==pfam]['prot-id|pfam'].values.tolist()
        # for protein in dark_proteins:
        protein_tokenized = prot_tokenizer.encode(undruggable[protein])
        # -----------------------------------------------------------------------------
        one_protein=pd.DataFrame()
        for j in range(bound):
            batch_drug_graph = graph_list[j*args['batchsize']:(j+1)*args['batchsize']]
            # batch_drug_name = pd.DataFrame(ikey_list[j*args['batchsize']:(j+1)*args['batchsize']])
            data={'drug-list':batch_drug_graph,'protein':protein_tokenized}
            prob = portal(data)
            batch_drug_name = pd.DataFrame(prob)
            batch_drug_name.columns =['NO', 'YES']
            one_protein= pd.concat([one_protein,batch_drug_name])

        batch_drug_graph = graph_list[bound * args['batchsize']:]
        # batch_drug_name = pd.DataFrame(ikey_list[bound * args['batchsize']:])
        data = {'drug-list': batch_drug_graph, 'protein': protein_tokenized}
        prob = portal(data)
        batch_drug_name = pd.DataFrame(prob)
        batch_drug_name.columns = ['NO', 'YES']
        one_protein = pd.concat([one_protein, batch_drug_name])
        # -----------------------------------------------------------------------------
        # screened_pfam[protein]= one_protein
        one_protein['uniprot|pfam']=protein

        screened = pd.concat([screened,one_protein])
        if i%args['step']==0:
            print('saved at:   --- ',checkpoint_path            )
            screened.to_pickle(checkpoint_path + 'undruggable_screen'+str(i*args['step'])+'.pkl')
            screened = pd.DataFrame()

        i+=1

screened.to_pickle(checkpoint_path + 'undruggable_screen'+'_TAIL'+'.pkl')
print('~ done ~')