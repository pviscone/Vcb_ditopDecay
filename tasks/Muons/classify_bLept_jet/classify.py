# %%
#!------------------Imports-------------------
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tqdm.notebook import tqdm

import MLP_model
import importlib
importlib.reload(MLP_model)
MLP = MLP_model.MLP

import utils
importlib.reload(utils)
OrderedDict = utils.OrderedDict
train=utils.train
plot_efficiency=utils.plot_efficiency


#%%
#!------------------Data loading-------------------
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("../Jet_features.pkl", compression="bz2")

# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
test_size=0.2


    
event_id = data_df["event_id"]
data_df = data_df.loc[:, data_df.columns != "event_id"]
_,event_id_test,= train_test_split(event_id,test_size=test_size,shuffle=False)



# %%
#!------------------N-1-------------------

nminus1_efficiency = []
col_removed = []


all_efficiency = train(data_df, label, event_id_test, test_size=test_size)

nminus1_efficiency.append(all_efficiency)
col_removed.append("All features")

err = 1/np.sqrt(test_size*len(data_df.groupby(level=0)))

n_col_loop = tqdm(range(len(data_df.columns)-1), desc="N-1 loop")
for n_col in n_col_loop:
    if n_col==0:
        data_nminus1 = data_df
    else:
        data_nminus1 = data_nminus1.loc[:,data_nminus1.columns != col_to_remove]

    dict_nminus1_step = OrderedDict({})
    column_loop = tqdm(data_nminus1.columns, desc="Column")
    for col in column_loop:
        data_nminus1_step = data_nminus1.loc[:,data_nminus1.columns != col]
        efficiency = train(data_nminus1_step,label,event_id_test,test_size=test_size)
        plt.title(f"N-{n_col+1}: {col}")
        plt.savefig(f"./images/N_minus1/loss/N-{n_col+1}_{col}.png")
        dict_nminus1_step[col] = efficiency

    dict_nminus1_step = dict_nminus1_step.sort()
    dict_nminus1_step = dict_nminus1_step.insert(0, ("All features", all_efficiency))

    plot_efficiency(dict_nminus1_step, err=err)
    if n_col > 0:
        plt.title(f"step: N-{n_col+1}. Previously removed: {col_to_remove}")
    else:
        plt.title(f"step: N-{n_col+1} (first step)")
    plt.savefig(f"./images/N_minus1/step_N-{n_col+1}.png")
    

    col_to_remove = dict_nminus1_step[-1][0]
    col_removed.append(f"N-{n_col+1}: {col_to_remove}")
    nminus1_efficiency.append(dict_nminus1_step[-1][1])

#!Remember that the efficiency of the last surviving feature is obtained with a training with 2 features
col_removed.insert(len(col_removed), f"N-{len(data_df.columns)-1}: {dict_nminus1_step[-2][0]}")
nminus1_efficiency.insert(len(nminus1_efficiency), dict_nminus1_step[-2][1])

dict_nminus1=OrderedDict(zip(col_removed, nminus1_efficiency))
plot_efficiency(dict_nminus1, err=err)
plt.title("N-1: All steps")
plt.savefig(f"./images/N_minus1/all_steps.png")



# %%
#!------------------N+1-------------------
err = 1/np.sqrt(test_size*len(data_df.groupby(level=0)))

nplus1_efficiency = []
col_added = []
col_list = list(data_df.keys())

initial_features=["Jet_CvBtag", "dPhi_Jet_mu"]
[col_list.remove(col) for col in initial_features]

data_nplus1 = data_df.loc[:, initial_features]

first_efficiency = train(data_nplus1, label,event_id_test,test_size=test_size)
nplus1_efficiency.append(first_efficiency)
col_added.append("(starter) CvB, dphi_j_mu")

n_col_loop = tqdm(range(len(col_list)), desc="N+1 loop")
for n_col in n_col_loop:
    if n_col >0:
        data_nplus1=data_nplus1.join(data_df.loc[:, col_to_add])

    dict_nplus1_step = OrderedDict({})
    column_loop = tqdm(col_list, desc="Column")
    for col in column_loop:
        data_nplus1_step = data_nplus1.join(data_df.loc[:, col])
        efficiency=train(data_nplus1_step,label,event_id_test,test_size=test_size)
        plt.title(f"N+{n_col+1}: {col}")
        plt.savefig(f"./images/N_plus1/loss/N+{n_col+1}_{col}.png")
        dict_nplus1_step[col] = efficiency

    dict_nplus1_step = dict_nplus1_step.sort()
    dict_nplus1_step = dict_nplus1_step.insert(0, ("(starter) CvB, dphi_jmu", first_efficiency))

    plot_efficiency(dict_nplus1_step, err=err)
    if n_col > 0:
        plt.title(f"step: N+{n_col+1}. Previously added: {col_to_add}")
    else:
        plt.title(f"step: N+{n_col+1} (first step)")
    plt.savefig(f"./images/N_plus1/step_N+{n_col+1}.png")
    

    col_to_add = dict_nplus1_step[-1][0]
    col_added.append(f"N+{n_col+1}: {col_to_add}")
    nplus1_efficiency.append(dict_nplus1_step[-1][1])
    col_list.remove(col_to_add)


dict_nplus1 = OrderedDict(zip(col_added, nplus1_efficiency))
plot_efficiency(dict_nplus1, err=err)
plt.title("N+1: All steps")
plt.savefig(f"./images/N_plus1/all_steps.png")

# %%
