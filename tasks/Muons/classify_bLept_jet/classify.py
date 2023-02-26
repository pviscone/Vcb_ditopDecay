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

#%%
# enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
dev="cpu"

device = torch.device(dev)



df = pd.read_pickle("./Jet_features.pkl", compression="bz2")
#%% Load data and create variables


event_id_df=df["event_id"]
label_df=df["label"].astype(float)
data_df=df.loc[:,~df.columns.isin(["label","event_id"])]

data=torch.tensor(data_df.values,device=device)
label=torch.tensor(label_df.values,device=device,dtype=torch.float32)

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.2,shuffle=True)

#%%
#!FIXME: I imposed to use the gpu. Fix the MLP clss to use the gpu if available

#!FIXME: I applyed some "unsqueeze" to the labels. Fix the MLP class to accept labels with shape (n_samples,1) (check)
importlib.reload(MLP_model)
MLP = MLP_model.MLP
model=MLP(x_train=train_data,y_train=train_label.unsqueeze(dim=1),x_test=test_data,y_test=test_label.unsqueeze(dim=1),learning_rate=0.001)

#%%

model.train_loop(epochs=1000)
# %%
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(model.train_loss,label="train")
plt.plot(model.test_loss,label="test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("BCE loss")
plt.subplot(122)
plt.plot(model.train_accuracy,label="train")
plt.plot(model.test_accuracy,label="test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
# %%
