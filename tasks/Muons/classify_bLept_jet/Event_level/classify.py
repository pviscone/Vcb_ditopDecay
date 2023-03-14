#%%
import sys
sys.path.append('..')

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

train_data,test_data,train_label,test_label=train_test_split(data,label, test_size=0.15, shuffle=True)

#%%

importlib.reload(MLP_model)
MLP = MLP_model.MLP
AttentionNetwork = MLP_model.AttentionNetwork


model =MLP(x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label,
            hidden_arch=[50,50,50,50],  batch_size=5000,
            optim={"lr": 0.0005, "weight_decay": 0.001, },
            early_stopping=None
            )
model = model.to(device)

#!---------------------Training---------------------

model.train_loop(epochs=500)

#!---------------------Plot loss---------------------
model.loss_plot()

# %%
"""
mask = (model.y_test.squeeze()) != model(model.x_test).argmax(axis=1)
x=model.x_test[mask].to(cpu).detach().numpy()
select=6*np.ones(x.shape[0],dtype=int)
plt.histnp.take_along_axis(x,select,axis=1),bins=100,range=(0,200))

"""