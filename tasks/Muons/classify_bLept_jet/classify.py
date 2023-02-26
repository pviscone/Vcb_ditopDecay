#%% Imports
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak


import MLP_model
import importlib
importlib.reload(MLP_model)
MLP=MLP_model.MLP

import pandas as pd


# enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)



df = pd.read_pickle("./Jet_features.pkl", compression="bz2")
#%% Load data and create variables


event_id_df=df["event_id"]
label_df=df["label"]
data_df=df.loc[:,~df.columns.isin(["label","event_id"])]

data=torch.tensor(data_df.values,device=device)
label=torch.tensor(label_df.values,device=device)

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.2,shuffle=True)

#%%
model=MLP(x_train=train_data,y_train=train_label,x_test=test_data,y_test=test_label,learning_rate=0.001)
