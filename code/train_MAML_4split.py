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
from DTI_meta_MAML_4split import *
# -------------------------------
parser = argparse.ArgumentParser("DTI in a MAML way: 4 split")
# ... # ...admin # ...# ...# ...
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--fr_scratch',type=str2bool, default=False)
parser.add_argument('--cwd', type=str,  default='',
                    help='define your own current working directory,i.e. where you put your scirpt')
# ... # ...meta core # ...# ...# ...
parser.add_argument('--k_spt', default=1, type=int, help='k-shot learning')
parser.add_argument('--k_qry', default=4,type=int)
parser.add_argument('--task_num',default=5,type=int)
parser.add_argument('--mix_n',default=1,type=int)
parser.add_argument('--meta_lr',default=1e-3,type=float)
parser.add_argument('--update_lr',default=0.01,type=float)
parser.add_argument('--update_step',default=5,type=int)
parser.add_argument('--update_step_test',default=10,type=int)
# ... # ...model # ...# ...# ...
parser.add_argument('--frozen',default='none',type=str)
parser.add_argument('--phase1ResiDist', type=str, default='none', help = {'multiply-binary','none'})
parser.add_argument('--phase2DTIDist', type=str, default='multiply-multi', help = {'multiply-binary','none'})
# ... # ...global optimization # ...# ...# ...
parser.add_argument('--global_step', default=30, type=int,
                    help='Number of global training steps, i.e. numberf of mini-batches ')
parser.add_argument('--global_eval_at', default=10, type=int, help='')
parser.add_argument('--global_eval_step',default=1000, type=int)
opt = parser.parse_args()
# -------------------------------
args = {}
args.update(vars(opt))
DTI_classifier_config ={
    'chem_pretrained':'nope',
    'protein_descriptor':'DISAE',
    'DISAE':{'frozen_list':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}
}
args.update(DTI_classifier_config)
seed = 705
np.random.seed(7)
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    args['device']='cuda'
else:
    args['device'] = 'cpu'
checkpoint_path = set_up_exp_folder('../results/',exp ='exp_4split_logs/')
config_file = checkpoint_path + 'config.json'
save_json(vars(opt),    config_file)
# -------------------------------
if __name__ == '__main__':
    # load data
    MAML_DTI_path = '../data/ChEMBLE26/interaction/for_MAML/'
    path =  '../data/exp_4split_logs/exp24-06-2021-13-04-30/'
    train_dict = load_pkl(MAML_DTI_path + 'train_4split_L5.pkl')
    iid_dev = pd.read_csv(MAML_DTI_path + 'IID_dev.csv')
    ood_dev = pd.read_csv(MAML_DTI_path + 'ood_dev.csv')
    ood_test = pd.read_csv(MAML_DTI_path + 'ood_test.csv')
    trainpfam = list(train_dict.keys())

    # set up meta core
    maml = meta_DTI_MAML(args)
    if args['fr_scratch']==False:
        print('loading trained model for a quick demo.... ')
        maml.model.load_state_dict(torch.load(path +'maml_model.dat'))

    train_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    iid_dev_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    ood_dev_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    ood_test_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    best_test_AUC = -np.inf
    best_test_AUPR = -np.inf
    stime = time.time()
    print('\tF1\tROC-AUC\tPR-AUC')
    for step in range(args['global_step']):
        batch_data = sample_minibatch_mixpfam(trainpfam, args['task_num'],args['mix_n'],args['k_spt'], args['k_qry'], train_dict)
        losses, overall,class0,class1 = maml(batch_data)
        train_performance['loss'].append(losses.detach().cpu().item())
        train_performance['overall'].append(overall)
        train_performance['class0'].append(class0)
        train_performance['class1'].append(class1)
        # ---------------
        # -------------------------------------------
        #         evaluating
        # -------------------------------------------
        if step % args['global_eval_at'] ==0:
            print('------------------------training step: ', step)

            # -------------------------------------------
            #         zero-shot test
            # -------------------------------------------
            overall_iid_dev, class0_iid_dev, class1_iid_dev, loss_iid_dev = [], [], [], []
            overall_ood_dev, class0_ood_dev, class1_ood_dev, loss_ood_dev = [], [], [], []
            overall_ood_test, class0_ood_test, class1_ood_test, loss_ood_test = [], [], [], []
            for step_eval in range(args['global_eval_step']):
                batch_data_iid_dev = iid_dev.sample(args['k_qry']*2*args['mix_n'])
                if len(set(batch_data_iid_dev['Activity'].values.tolist()))>1:
                    losses, overall, class0, class1 = maml.zeroshot_test(batch_data_iid_dev)
                    overall_iid_dev.append(overall)
                    class0_iid_dev.append(class0)
                    class1_iid_dev.append(class1)
                    loss_iid_dev.append(losses.detach().cpu().item())
                batch_data_ood_dev = ood_dev.sample(args['k_qry'] * 2 * args['mix_n'])
                if len(set(batch_data_ood_dev['Activity'].values.tolist())) > 1:
                    losses, overall, class0, class1 = maml.zeroshot_test(batch_data_ood_dev)
                    overall_ood_dev.append(overall)
                    class0_ood_dev.append(class0)
                    class1_ood_dev.append(class1)
                    loss_ood_dev.append(losses.detach().cpu().item())
                batch_data_ood_test = ood_test.sample(args['k_qry'] * 2 * args['mix_n'])
                if len(set(batch_data_ood_test['Activity'].values.tolist())) > 1:
                    losses, overall, class0, class1 = maml.zeroshot_test(batch_data_ood_test)
                    overall_ood_test.append(overall)
                    class0_ood_test.append(class0)
                    class1_ood_test.append(class1)
                    loss_ood_test.append(losses.detach().cpu().item())
            iid_dev_performance['loss'].append(np.mean(loss_iid_dev))
            iid_dev_performance['overall'].append(np.array(overall_iid_dev).mean(axis=0))
            iid_dev_performance['class0'].append(np.array(class0_iid_dev).mean(axis=0))
            iid_dev_performance['class1'].append(np.array(class1_iid_dev).mean(axis=0))

            ood_dev_performance['loss'].append(np.mean(loss_ood_dev))
            ood_dev_performance['overall'].append(np.array(overall_ood_dev).mean(axis=0))
            ood_dev_performance['class0'].append(np.array(class0_ood_dev).mean(axis=0))
            ood_dev_performance['class1'].append(np.array(class1_ood_dev).mean(axis=0))

            ood_test_performance['loss'].append(np.mean(loss_ood_test))
            ood_test_performance['overall'].append(np.array(overall_ood_test).mean(axis=0))
            ood_test_performance['class0'].append(np.array(class0_ood_test).mean(axis=0))
            ood_test_performance['class1'].append(np.array(class1_ood_test).mean(axis=0))


            # -------------------------------------------
            #         save records
            # -------------------------------------------

            save_dict_pickle(train_performance,checkpoint_path  +'train_performance.pkl')
            save_dict_pickle(iid_dev_performance, checkpoint_path + 'iid_dev_performance.pkl')
            save_dict_pickle(ood_dev_performance, checkpoint_path + 'ood_dev_performance.pkl')
            save_dict_pickle(ood_test_performance, checkpoint_path + 'ood_test_performance.pkl')
            print('train\t', train_performance['overall'][-1])
            print('iid_dev\t', iid_dev_performance['overall'][-1])
            print('ood_dev\t', ood_dev_performance['overall'][-1])
            print('ood_test\t', ood_test_performance['overall'][-1])
            # print('time cost of the episode: ', time.time() - stime)
            stime = time.time()
            # if ood_dev_performance['overall'][-1][1] > best_test_AUC :
            #     best_test_AUC = ood_dev_performance['overall'][-1][1]
                # torch.save(maml.model.state_dict(), checkpoint_path+'maml_model.dat')
                # print('..................saved at:', checkpoint_path)
            # elif ood_dev_performance['overall'][-1][2] > best_test_AUPR:
            #     best_test_AUPR =ood_dev_performance['overall'][-1][2]
            #     torch.save(maml.model.state_dict(), checkpoint_path + 'maml_model.dat')
            #     print('..................saved at:', checkpoint_path)

print('training finished ~')
print(f'model performance record saved to {checkpoint_path}')