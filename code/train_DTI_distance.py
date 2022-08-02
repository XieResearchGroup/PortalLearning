# Protein inter-residue distance prediction
# dat: PDB binding-site specific
# protein descriptor: TAPE or DISAE
# ------------- admin
import warnings
warnings.filterwarnings('ignore')
import argparse

import time
import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
# ------------------------
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
# -------------  my work
from model_Yang import *
from ligand_graph_features import *
from data_tool_box import *
from models_plusResnet import *
from model_piple_utils import *
from utils import *
#-------------------------------------------
#      set hyperparameters
#-------------------------------------------
parser = argparse.ArgumentParser("DTI distance prediction")
# ---------- args for admin
parser.add_argument('--cwd', type=str, default='')
parser.add_argument('--distance_path', type=str, default='../data/distance/')
#---------- args for distance prediction
parser.add_argument('--pred_mode', type=str, default='binary',help='choose from [binary, multi ]')
parser.add_argument('--feat_mode', type=str, default='multiply',help='choose from [attentive-pool, multiply ]')
#---------- args for protein descriptor
parser.add_argument('--protein_descriptor', type=str, default='DISAE')
parser.add_argument('--frozen', type=str, default='whole',help='choose from {whole, none,partial}')
parser.add_argument('--chem_pretrained', type=str, default='nope',help='choose from {YES,nope}')
parser.add_argument('--prot_frPhase1', type=str, default='nope',help='choose from {YES,nope}')
parser.add_argument('--frDTI', type=str, default='none',help='choose from {none,DISAE-DTI}')
####---------- args for model training and optimization
parser.add_argument('--global_step', default=200, type=int, help='Number of training epoches ')
parser.add_argument('--eval_at', default=10, type=int, help='')
parser.add_argument('--batch_size', default=45, type=int, help="Batch size")
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")
parser.add_argument('--multigpu', type=str, default='', help='multi gpu')
#----------
opt = parser.parse_args()
# ------------- protein descriptor

# print('using DISAE+')
from transformers import BertTokenizer
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertForMaskedLM
from transformers.modeling_albert import load_tf_weights_in_albert

# -------------------------------------------
#         set admin
# -------------------------------------------
seed = 705
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
checkpoint_dir = set_up_exp_folder('../results/',exp ='exp_logs_DTIdist/')
config_file = checkpoint_dir + 'config.json'
save_json(vars(opt),    config_file)


def evaluate_at_batch(trainset,batch_size,tokenizer,device,model,pred_mode,protein_descriptorChoice,chem_pdb_file_path):
    trainset_size=len(trainset)
    eval_train_at = np.random.randint(trainset_size-batch_size)
    batch_data = trainset[eval_train_at:eval_train_at+batch_size]
    # print(len(batch_data))
    batch_input,unpad_targets,sizes = generate_batch_input_DTIdist(batch_data, batch_size,
                                                                   tokenizer,pred_mode,
                                                                   protein_descriptorChoice,
                                                                   chem_pdb_file_path,
                                                                   device)
    logits = model(batch_input,device)
    batch_acc= []
    for i in range(batch_size):
        l =  logits[i,:sizes[i][0],:sizes[i][1],:].detach().cpu()
        p =  torch.argmax(l,axis=2)
        acc = torch.sum(p==unpad_targets[i])/(sizes[i][0]*sizes[i][1])
        batch_acc.append(acc)

    return np.mean(batch_acc)

#-------------------------------------------
#         main
#-------------------------------------------
if __name__ == '__main__':
    if opt.multigpu!='':
        opt.multigpu = [int(f) for f in opt.multigpu.split(',')]
    if opt.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        if len(opt.multigpu)>1:
            device=torch.device('cuda:'+str(opt.multigpu[0]))
    else:
        device = torch.device('cpu')
    ligand_path =  opt.distance_path + 'ligand-related/'
    chem_pdb_file_path = ligand_path + 'ghose/'
    # -------------------------------------------
    #         set up model
    # -------------------------------------------


    dim = 312
    # -------------------related hyperparameter-----
    albertdatapath = '../data/albertdata/DISAE_plus/'
    albertvocab = os.path.join( albertdatapath, 'pfam_vocab_triplets.txt')
    albertconfig = os.path.join(  albertdatapath, 'albert_config_tiny_google.json')
    albert_pretrained_checkpoint = os.path.join( albertdatapath, "model.ckpt-3000000")
    # ------------------real thing------
    berttokenizer = BertTokenizer.from_pretrained(albertvocab)
    albertconfig = AlbertConfig.from_pretrained(albertconfig)
    model_albert = AlbertForMaskedLM(config=albertconfig)
    model_albert = load_tf_weights_in_albert(model_albert, albertconfig, albert_pretrained_checkpoint)
    albert = model_albert.albert
    protein_descriptor = albert
    tokenizer = berttokenizer
    chem_descriptor = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')

    model = DTI_distMtx_classifier(dim=dim, feat_mode=opt.feat_mode,
                                            pred_mode=opt.pred_mode,
                                            protein_descriptor=protein_descriptor,
                                   chem_descriptor=chem_descriptor,
                                   frozen=opt.frozen,
                                   cwd ='',chem_pretrained=opt.chem_pretrained)
    if len(opt.multigpu)>0:
        model=torch.nn.DataParallel(model,device_ids=opt.multigpu)
        print('model sent to multi gpu:', opt.multigpu)

    model = model.to(device)
    # -------------------------------------------
    #         set up optimization
    # -------------------------------------------
    loss_fn = nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=opt.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # -------------------------------------------
    #         set up data
    # -------------------------------------------
    trainset = load_pkl(opt.distance_path  + 'residue-ligand/' + 'train.pkl')
    testset =  load_pkl(opt.distance_path  + 'residue-ligand/' + 'test.pkl')
    trainset_size = len(trainset)

    # -------------------------------------------
    #         training
    # -------------------------------------------
    bes_test_acc = -np.inf
    best_epoch = 0
    loss_train=[]
    train_acc_by_epoch , test_acc_by_epoch =[],[]
    print("Dataset\tMSE")
    stime = time.time()
    for step in range(opt.global_step):
        model.train()
        start = np.random.randint(trainset_size-opt.batch_size)
        batch_data = trainset[start: start + opt.batch_size]
        batch_input,unpad_targets,sizes = generate_batch_input_DTIdist(batch_data,opt.batch_size,
                                                                       tokenizer,opt.pred_mode,
                                                                       opt.protein_descriptor,
                                                                       chem_pdb_file_path,
                                                                       device)
        logits = model(batch_input,device)
        logits_masked = logits * batch_input['targets-mask'][:, :, :, None] # mask for batch padding
        logits_mv = logits_masked.permute(0, 3, 1, 2)
        loss = loss_fn(logits_mv,batch_input['targets-padded'])
        # ----as always----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #------------------
        loss_train.append(loss.detach().cpu().numpy())
        # -------------------------------------------
        #         evaluating
        # -------------------------------------------
        if step % opt.eval_at == 0:
            print('------------------------global step: ', step)
            model.eval()
            batch_acc_train,batch_acc_test=[],[]
            for i in range(5):
                acc_train = evaluate_at_batch(trainset,opt.batch_size,tokenizer,  device,
                                              model,opt.pred_mode,
                                              opt.protein_descriptor,
                                              chem_pdb_file_path)
                batch_acc_train.append(acc_train)
                acc_test = evaluate_at_batch(testset, opt.batch_size, tokenizer, device,
                                             model, opt.pred_mode,
                                              opt.protein_descriptor,
                                              chem_pdb_file_path)
                batch_acc_test.append(acc_test)

            train_acc_by_epoch.append(np.mean(batch_acc_train))
            test_acc_by_epoch.append(np.mean(batch_acc_test))
            print('train\t',np.mean(batch_acc_train))
            print('test\t', np.mean(batch_acc_test))

            # -------------------------------------------
            #         save records
            # -------------------------------------------
            np.save(checkpoint_dir+'loss_train.npy',loss_train)
            np.save(checkpoint_dir+'train_acc.npy',train_acc_by_epoch)
            np.save(checkpoint_dir+'test_acc.npy',test_acc_by_epoch)
            print('time cost of the episode: ', time.time() - stime)
            print('                                  stored at: ',checkpoint_dir)
            stime = time.time()
            if np.mean(batch_acc_test) > bes_test_acc:
                bes_test_acc = np.mean(batch_acc_test)
                best_epoch=step
                torch.save(model.state_dict(),checkpoint_dir+'model.dat')

