#%%
import sys
sys.path.append('..')


import utils
import importlib
import MLP_model
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt



importlib.reload(MLP_model)
MLP = MLP_model.MLP

importlib.reload(utils)
OrderedDict = utils.OrderedDict
train = utils.train
plot_efficiency = utils.plot_efficiency


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("./event_df.pkl", compression="bz2")



label=np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
df=df.loc[:, df.columns != "label"]
df=df.drop(columns=["Muon_mass", "Neutrino_mass"])


"""
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()
"""

data = torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)
label = torch.tensor(label, dtype=torch.long, device=device)

train_data,test_data,train_label,test_label=train_test_split(data,label, test_size=0.2, shuffle=True)

#%%

importlib.reload(MLP_model)
MLP = MLP_model.MLP


model = MLP(x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label,
            hidden_arch=[20,20,20], batch_size=1000,
            optim={"lr": 0.0005, "weight_decay": 0.001, },
            early_stopping=None
            )
model = model.to(device)

#! SOMETHIMG IS BROKEN
#!---------------------Training---------------------

model.train_loop(epochs=100)

#!---------------------Plot loss---------------------
model.loss_plot()

# %%
