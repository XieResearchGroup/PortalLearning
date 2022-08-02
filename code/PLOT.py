import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_pkl(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return  dict

def load_data_meta(path):
    train_p = load_pkl(path +'train_performance.pkl')
    iid_dev_p = load_pkl(path +'iid_dev_performance.pkl')
    ood_dev_p = load_pkl(path+'ood_dev_performance.pkl')
    ood_test_p = load_pkl(path+'ood_test_performance.pkl')
    return [train_p,iid_dev_p,ood_dev_p,ood_test_p]

def load_data_meta3split(path):
    train_p = load_pkl(path +'train_performance.pkl')
    ood_test_p = load_pkl(path +'test_performance.pkl')
    iid_dev_p = load_pkl(path+'dev_performance.pkl')
    return [train_p,iid_dev_p,'a',ood_test_p]

def cal_rolling_metrics_meta(data,window,j):
    df = pd.DataFrame(data).iloc[:,j]
    mu = df.rolling(window=window).mean()
    st = df.rolling(window=window).std()
    upper = mu+st
    lower = mu-st
    return mu,upper,lower

cwd = '../data/exp_4split_logs/'
path = cwd  +'exp24-06-2021-13-04-30/'
data4split = load_data_meta(path)
baseline = load_data_meta3split( cwd +'[baseline with breakdown] exp09-06-2021-19-21-59/')
data = [data4split,baseline]
exp_names=['portal learning','baseline']
colors=['lime','orange']

trainwindow=100 # every 100 training steps is one epoch
gap_4split = pd.DataFrame(data4split[3]['overall']).rolling(trainwindow).mean() - pd.DataFrame(data4split[2]['overall']).rolling(trainwindow).mean()
gap_3split = pd.DataFrame(baseline[3]['overall']).rolling(trainwindow).mean() - pd.DataFrame(baseline[1]['overall']).rolling(trainwindow).mean()

fig, axs = plt.subplots(1, 3, figsize=(3.5 * 3, 3.2 * 1), sharey=True)

i = 0
for j in range(3):  # number of metrics#     for k in range(4): # number of split
    axs[j].plot(gap_4split.iloc[:, j], c=colors[i], label=exp_names[i])

i = 1
for j in range(3):  # number of metrics#     for k in range(4): # number of split
    axs[j].plot(gap_3split.iloc[:, j], c=colors[i], label=exp_names[i])
axs[0].set_ylabel('deployment gap')
axs[0].set_title('F1 score ')
axs[1].set_title('ROC-AUC ')
axs[2].set_title('PR-AUC')

plt.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),
           frameon=False)
fig.suptitle('Deployment gap: test perfromance - dev performance', y=1.02)
# plt.show()
fig.savefig('../results/deploymentgap.png')

key = 'overall'
xlim_train=800
xlim_test=800
trainwindow=100
testwindow=100
std=False

fig, axs = plt.subplots(4, 3, figsize=(4*3,3*4),sharey=True)
windows = [trainwindow,testwindow,testwindow,testwindow]

i= 0
for j in range(3): # number of metrics
    for k in range(4): # number of split
        mu0,upper0,lower0 = cal_rolling_metrics_meta(data[i][k][key],windows[k],j)
        x = list(range(len(mu0)))
        axs[k,j].plot(x,mu0,c= colors[i],label=exp_names[i])
        if std==True:
            axs[k,j].fill_between(x,lower0,upper0,facecolor=colors[i],alpha=0.1)

    axs[0,j].set_xlim(trainwindow,xlim_train)
    axs[1,j].set_xlim(testwindow,xlim_test)
    axs[2,j].set_xlim(testwindow,xlim_test)
    axs[3,j].set_xlim(testwindow,xlim_test)

i=1
0
for j in range(3): # number of metrics
    for k in [0,1,3]: # number of split
        mu0,upper0,lower0 = cal_rolling_metrics_meta(data[i][k][key],windows[k],j)
        x = list(range(len(mu0)))
        axs[k,j].plot(x,mu0,c= colors[i],label=exp_names[i])
        if std==True:
            axs[k,j].fill_between(x,lower0,upper0,facecolor=colors[i],alpha=0.1)

    axs[0,j].set_xlim(trainwindow,xlim_train)
    axs[1,j].set_xlim(testwindow,xlim_test)
    axs[2,j].set_xlim(testwindow,xlim_test)
    axs[3,j].set_xlim(testwindow,xlim_test)


axs[0,0].set_ylabel('TRAN')
axs[1,0].set_ylabel('iid-DEV')
axs[2,0].set_ylabel('OOD-DEV')
axs[3,0].set_ylabel('OOD-TEST')
axs[0,0].set_title('F1 score ')
axs[0,1].set_title('ROC-AUC ')
axs[0,2].set_title('PR-AUC')


plt.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),
               frameon=False)
# plt.show()
fig.savefig('../results/4split-training-curve.png')
