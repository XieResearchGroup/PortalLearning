import torch
from torch import  nn
from torch import optim
from torch.nn import functional as F
from copy import deepcopy
# -------------
from data_tool_box import *
from utils_MAML import *
# -------------
from models_MAML import  *
from models_pipeline import *
from model_piple_utils import *
from model_Yang import *

#=============================================
#  after incorporating Di's idea
class meta_DTI_MAML(nn.Module):
    def __init__(self,args):
        super(meta_DTI_MAML, self).__init__()
        # .......... LOAD DISAE ..........
        prot_descriptor, self.prot_tokenizer = load_DISAE(args['cwd'])
        self.chem_dict = pd.Series(load_json(args['cwd']+'data/ChEMBLE26/chemical/ikey2smiles_ChEMBLE.json'))
        self.protein_dict = pd.Series(load_pkl(args['cwd']+'data/ChEMBLE26/protein/unipfam2triplet.pkl'))

        # .......... set up DTI model ..........
        # -------------------------------------------
        #         distance pretraining
        # -------------------------------------------
        dim=312
        all_config=args
        protein_descriptor = prot_descriptor
        if all_config['phase1ResiDist'] != 'none':
            print('loading protein descriptor trained on inter-residue distance prediction')
            model_interResiDist = interResidue_distMtx_classifier(dim=dim,
                                                                  feat_mode=all_config['phase1ResiDist'].split('-')[0],
                                                                  pred_mode=all_config['phase1ResiDist'].split('-')[1],
                                                                  protein_descriptor=protein_descriptor,
                                                                  frozen='none')
            path = args['cwd'] + 'data/distance/residue-residue/model-chosen/' + 'DISAE' + '-' + 'none' + '-' + \
                   all_config['phase1ResiDist']
            model_interResiDist.load_state_dict(torch.load(path + '/model.dat'))
            for name, m in model_interResiDist.named_children():
                print(name)
                if name == 'protein_descriptor':
                    protein_descriptor0 = m
            chem_descriptor0 = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')
        if all_config['phase2DTIDist'] != 'none':
            print('load back DTI distance prediction protein/chem descriptors')
            model_DTIDist = DTI_distMtx_classifier0(dim=dim,
                                                    feat_mode=all_config['phase2DTIDist'].split('-')[0],
                                                    pred_mode=all_config['phase2DTIDist'].split('-')[1],
                                                    protein_descriptor=protein_descriptor,
                                                    frozen='none',
                                                    cwd=args['cwd'], chem_pretrained='nope')
            path = args['cwd'] + 'data/distance/residue-ligand/model-chosen/' + 'DISAE' + '-' + 'none' + '-' + \
                   all_config['phase2DTIDist']
            model_DTIDist.load_state_dict(torch.load(path + '/model.dat'))
            for name, m in model_DTIDist.named_children():
                print(name)
                if name == 'protein_descriptor':
                    protein_descriptor0 = m
                if name == 'chem_decriptor':  # HERITAGE TYPO
                    chem_descriptor0 = m
                if name == 'prot_transform':
                    prot_transform2 = m
        if all_config['phase1ResiDist'] == 'none' and all_config['phase2DTIDist'] == 'none':
            protein_descriptor0 = protein_descriptor
            chem_descriptor0 = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type='gin')
        self.model = DTI_model_MAML(all_config =args,model = protein_descriptor0,chem=chem_descriptor0).to(args['device'])
        self.meta_optim = optim.Adam(self.model.parameters(),lr = args['meta_lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.meta_optim, T_max=10)
        self.args = args
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self,batch_data):

        task_num = len(batch_data)
        losses_q_alltasks = 0
        overall_all_test, class0_all_test, class1_all_test = [],[],[]

        # ---- mete update crosss tasks
        self.model.zero_grad()
        self.model.train()
        model_weights_meta_init = list(self.model.parameters())
        for t in range(task_num):

        # =========================core======================
            batch_data_pertask = batch_data[t]
            # --- -------------------support set updating--
            for s in range(self.args['update_step']):

                logits, y_spt, loss = core_batch_prediction_MAML(batch_data_pertask, self.args,
                                                                 self.prot_tokenizer, 'spt',
                                                                 self.loss_fn,
                                                                 self.chem_dict,
                                                                 self.protein_dict,
                                                                 self.model, detach=False)
                grad = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)

                fast_weights = []
                for p in zip(grad, self.model.parameters()):
                    if type(p[0]) != type(None):
                        fast_weights.append(p[1] - self.args['update_lr'] * p[0])
                    else:
                        fast_weights.append(torch.zeros_like(p[1]))

                for param_meta, param_update in zip(self.model.parameters(), fast_weights):
                    param_meta.data = param_update.data
            # --- -----------------query set get evaluated----
            logits_q, y_qry, loss_q = core_batch_prediction_MAML(batch_data_pertask, self.args,
                                                             self.prot_tokenizer, 'qry',
                                                                 self.loss_fn,
                                                                 self.chem_dict,
                                                                 self.protein_dict,
                                                                 self.model,  detach=True)


            losses_q_alltasks += loss_q
            overall,class0,class1 = evaluate_binary_predictions(y_qry, logits_q)
            overall_all_test.append(overall)
            class0_all_test.append(class0)
            class1_all_test.append(class1)

            # reset model param for next task
            for param_meta, param_init in zip(self.model.parameters(), model_weights_meta_init):
                param_meta.data = param_init.data
        # =========================core======================


        # ---- mete update crosss tasks
        losses_q_metaupdate = losses_q_alltasks/self.args['task_num']
        self.meta_optim.zero_grad()
        losses_q_metaupdate.backward() #need to check what will be triggered by this
        self.meta_optim.step()
        self.scheduler.step()
        return losses_q_metaupdate,np.mean(overall_all_test,axis=0),np.mean(class0_all_test,axis=0),np.mean(class1_all_test,axis=0)

    def finetuning(self,batch_data):
        model_finetuning = deepcopy(self.model)

        # =========================core======================
        batch_data_pertask = batch_data[0]
        # --- ---------------------
        for s in range(self.args['update_step_test']):
            # print('update step: ', s)
            logits, y_spt, loss = core_batch_prediction_MAML(batch_data_pertask, self.args,
                                                             self.prot_tokenizer, 'spt',
                                                             self.loss_fn,
                                                             self.chem_dict,
                                                             self.protein_dict,
                                                             model_finetuning, detach=False)
            grad = torch.autograd.grad(loss,model_finetuning.parameters(), allow_unused=True)

            fast_weights = []
            for p in zip(grad,model_finetuning.parameters()):
                if type(p[0]) != type(None):
                    fast_weights.append(p[1] - self.args['update_lr'] * p[0])
                else:
                    fast_weights.append(torch.zeros_like(p[1]))

            for param_meta, param_update in zip(model_finetuning.parameters(), fast_weights):
                param_meta.data = param_update.data
        # --- ---------------------
        logits_q, y_qry, loss_q = core_batch_prediction_MAML(batch_data_pertask, self.args,
                                                             self.prot_tokenizer, 'qry',
                                                             self.loss_fn,
                                                             self.chem_dict,
                                                             self.protein_dict,
                                                             model_finetuning, detach=True)

        overall,class0,class1 = evaluate_binary_predictions(y_qry, logits_q)
        del model_finetuning
        return loss_q, overall,class0,class1
    def zeroshot_test(self,batch_data):
        self.model.eval()
        logits_q, y_qry, loss_q = core_batch_prediction_0shot(batch_data, self.args,
                                                             self.prot_tokenizer,
                                                             self.loss_fn,
                                                             self.chem_dict,
                                                             self.protein_dict,
                                                             self.model, detach=True)

        overall, class0, class1 = evaluate_binary_predictions(y_qry, logits_q)
        return loss_q, overall,class0,class1
    def test_AUC(self, batch_data):
        self.model.eval()
        logits_q, y_qry, loss_q = core_batch_prediction_0shot(batch_data, self.args,
                                                              self.prot_tokenizer,
                                                              self.loss_fn,
                                                              self.chem_dict,
                                                              self.protein_dict,
                                                              self.model, detach=True)
        return logits_q,y_qry