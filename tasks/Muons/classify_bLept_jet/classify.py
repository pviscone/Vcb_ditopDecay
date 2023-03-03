#%% Imports
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import MLP_model
import importlib
importlib.reload(MLP_model)
MLP=MLP_model.MLP

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("./Jet_features.pkl", compression="bz2")

# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
#%% #! -------------- test/train data-------------------

train_data_df, test_data_df, train_label, test_label = train_test_split(
    data_df, label, test_size=0.2,shuffle=False)

event_id_train=train_data_df["event_id"]
event_id_test=test_data_df["event_id"]

train_label = torch.tensor(train_label, device=device, dtype=torch.float32)
test_label = torch.tensor(test_label, device=device, dtype=torch.float32)

train_data=train_data_df.loc[:, train_data_df.columns != "event_id"]
test_data=test_data_df.loc[:, test_data_df.columns != "event_id"]

dict_n_minus1={}
for col in train_data.columns:
    train_data = train_data.loc[:, train_data.columns != col]
    test_data = test_data.loc[:, test_data.columns != col]

    #To tensor
    train_data = torch.tensor(train_data.values, device=device)
    test_data = torch.tensor(test_data.values, device=device)

    #!---------------------Model definition---------------------
    importlib.reload(MLP_model)
    MLP = MLP_model.MLP

    model=MLP(x_train=train_data,y_train=train_label,x_test=test_data,y_test=test_label,
              hidden_arch=[1000,1000,1000],batch_size=20000,
              optim={"lr":0.001,"weight_decay":0.00001,}
              )
    model=model.to(device)

    #!---------------------Training---------------------
    model.train_loop(epochs=300)

    #!---------------------Plot loss---------------------
    model.loss_plot()
    plt.title(col)

    #!------------------Efficiency on events-------------------
    efficiency=model.evaluate_on_events()
    dict_n_minus1[col]=efficiency
