import pandas as pd
import random
from sklearn import metrics
# --------------
from transformers import BertTokenizer
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertForMaskedLM
from transformers.modeling_albert import load_tf_weights_in_albert
# --------------
from data_tool_box import *
# =======================================================
#             data related
# =======================================================
def sample_balanced(s,k_spt,pfam_sampled,train_dict):
    neg = train_dict[pfam_sampled]['neg_'+s].sample(k_spt,replace=True)
    pos = train_dict[pfam_sampled]['pos_'+s].sample(k_spt,replace=True)
    spt = pd.concat((neg,pos))
    return spt.sample(frac=1)

def sample_minibatch(trainpfam,task_num,k_spt,k_qry,train_dict):
    batch_pfam_idx = random.sample(range(len(trainpfam)), k = task_num)
    batch = []
    for idx in batch_pfam_idx:
        pfam_sampled = trainpfam[idx]
        spt =  sample_balanced('spt',k_spt,pfam_sampled,train_dict)
        qry =  sample_balanced('qry',k_qry,pfam_sampled,train_dict)
        batch.append({'spt':spt,'qry':qry})
    return batch


def mix_npfam(trainpfam, mix_n, k_spt, k_qry, train_dict):
    batch_pfam_idx = random.sample(range(len(trainpfam)), k=mix_n)
    spts = []
    qrys = []
    for idx in batch_pfam_idx:
        pfam_sampled = trainpfam[idx]
        spt = sample_balanced('spt', k_spt, pfam_sampled, train_dict)
        qry = sample_balanced('qry', k_qry, pfam_sampled, train_dict)
        spts.append(spt)
        qrys.append(qry)
    #         batch.append({'spt':spt,'qry':qry})
    SPT = pd.concat(spts).sample(frac=1)
    QRY = pd.concat(qrys).sample(frac=1)
    mixset = {}
    mixset['spt'] = SPT
    mixset['qry'] = QRY

    return mixset

def sample_minibatch_mixpfam(trainpfam,task_num,mix_n,k_spt,k_qry,train_dict):
    batch =[]
    for i in range(task_num):
        mixset = mix_npfam(trainpfam,mix_n,k_spt,k_qry,train_dict)
        batch.append(mixset)
    return batch
# =======================================================
#             model related
# =======================================================
def load_DISAE(cwd):
    all_config = load_json( 'DTI_config.json')
    albertconfig = AlbertConfig.from_pretrained('../'+ all_config['DISAE']['albertconfig'])
    m = AlbertForMaskedLM(config=albertconfig)
    m = load_tf_weights_in_albert(m, albertconfig, '../'+all_config['DISAE']['albert_pretrained_checkpoint'])
    prot_descriptor = m.albert
    prot_tokenizer = BertTokenizer.from_pretrained('../'+all_config['DISAE']['albertvocab'])
    return  prot_descriptor,prot_tokenizer

def get_repr_DTI_MAML(batch_data_pertask,mode,prot_tokenizer,chem_dict,protein_dict,args):
    batch_chem_graphs_spt,batch_protein_tokenized_spt = get_repr_DTI(batch_data_pertask[mode],
                                                                    prot_tokenizer,
                                                                    chem_dict,protein_dict,
                                                                    'DISAE')
    batch_chem_graphs_spt = batch_chem_graphs_spt.to(args['device'])
    batch_protein_tokenized_spt = batch_protein_tokenized_spt.to(args['device'])

    y_spt = torch.LongTensor(list(batch_data_pertask[mode]['Activity'].values)).to(args['device'])
    return batch_chem_graphs_spt,batch_protein_tokenized_spt,y_spt

def core_batch_prediction_MAML(batch_data_pertask, args, tokenizer,mode,loss_fn,
                               chem_dict, protein_dict, model,detach=False):

    batch_chem_graphs_spt, batch_protein_tokenized_spt, y_spt = get_repr_DTI_MAML(batch_data_pertask,
                                                              mode,
                                                              tokenizer, chem_dict, protein_dict,
                                                              args)

    batch_logits = model(batch_protein_tokenized_spt, batch_chem_graphs_spt)
    loss = loss_fn(batch_logits,y_spt)
    if detach == True:
        batch_logits = batch_logits.detach().cpu()
        y_spt = y_spt.detach().cpu()

    return batch_logits, y_spt,loss
def get_repr_DTI_0shot(batch_data,tokenizer,chem_dict,protein_dict,args):
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
    # if prot_descriptor_choice =='DISAE':
    uniprot_list = batch_data['uniprot+pfam'].values.tolist()
    protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])
    #  . . . .  to return  . . . .
    chem_graphs = chem_graphs.to(args['device'])
    protein_tokenized = protein_tokenized.to(args['device'])
    label = torch.LongTensor(list(batch_data['Activity'].values)).to(args['device'])
    return chem_graphs, protein_tokenized,label

def core_batch_prediction_0shot(batch_data_pertask, args, tokenizer,loss_fn,
                               chem_dict, protein_dict, model,detach=True):
    batch_chem_graphs_spt, batch_protein_tokenized_spt, y_spt = get_repr_DTI_0shot(batch_data_pertask,
                                                              tokenizer, chem_dict, protein_dict,
                                                              args)

    batch_logits = model(batch_protein_tokenized_spt, batch_chem_graphs_spt)
    loss = loss_fn(batch_logits,y_spt)
    if detach == True:
        batch_logits = batch_logits.detach().cpu()
        y_spt = y_spt.detach().cpu()
    return batch_logits, y_spt,loss

def evaluate_binary_predictions(label, predprobs):
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)
    # --------------------------by label---
    bothF1 = metrics.f1_score(label, predclass, average=None)
    bothprecision = metrics.precision_score(label, predclass, average=None)
    bothrecall = metrics.recall_score(label, predclass, average=None)
    class0 = [bothF1[0], bothprecision[0], bothrecall[0]]
    class1 = [bothF1[1], bothprecision[1], bothrecall[1]]
    # -------------------------overall---
    f1 = metrics.f1_score(label, predclass, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 1], pos_label=1)
    aupr = metrics.auc(reca, prec)

    overall = [f1,auc,aupr]
    return overall,class0,class1


