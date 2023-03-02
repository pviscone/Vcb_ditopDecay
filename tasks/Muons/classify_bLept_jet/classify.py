#%% Imports
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
#import awkward as ak


import MLP_model
import importlib
importlib.reload(MLP_model)
MLP=MLP_model.MLP

import pandas as pd
#import wandb
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
#%%
# enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)




df = pd.read_pickle("./Jet_features.pkl", compression="bz2")
#%% Load data and create variables


#event_id_df=df["event_id"]
label=np.expand_dims(df["label"].astype(float).to_numpy(),axis=1)
ohe = OneHotEncoder()
ohe.fit(label)

# [a,b]: a=non leptonic, b=leptonic
label=ohe.transform(label).toarray()


data_df = df.loc[:, df.columns != "label"]

train_data_df, test_data_df, train_label, test_label = train_test_split(
    data_df, label, test_size=0.2,shuffle=False)

event_id_train=train_data_df["event_id"]
event_id_test=test_data_df["event_id"]


train_data = torch.tensor(
    train_data_df.loc[:, train_data_df.columns != "event_id"].values, device=device)

test_data = torch.tensor(
    test_data_df.loc[:, test_data_df.columns != "event_id"].values, device=device)


train_label=torch.tensor(train_label,device=device,dtype=torch.float32)

test_label = torch.tensor(
    test_label, device=device, dtype=torch.float32)



#%%

importlib.reload(MLP_model)
MLP = MLP_model.MLP

model=MLP(x_train=train_data,y_train=train_label,x_test=test_data,y_test=test_label,hidden_arch=[300,300,300],
          batch_size=100000,
          shuffle=False,
          optim={"lr":0.001,
                  "weight_decay":0.001,
              }
          )

model=model.to(device)

#model.wandb_init(project="leptonic_jet_classification", config={"architecture": "MLP", "loss": "BCE", "optimizer": "RMSprop"})

#%%
model.train_loop(epochs=120)


# %%
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.plot(model.train_loss,label="train")
plt.plot(model.test_loss,label="test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("BCE loss")
plt.subplot(132)
plt.plot(model.false_negative,label="false negative")
plt.plot(model.false_positive,label="false positive")
plt.title("Type I and II error: TEST dataset")
plt.legend()
plt.xlabel("Epochs")
plt.subplot(133)
with torch.no_grad():
  confusion_matrix_test = torch.tensor(
      [[1-model.false_positive[-1], model.false_positive[-1]], [model.false_negative[-1], 1-model.false_negative[-1]]])

  confusion_matrix_test = pd.DataFrame(confusion_matrix_test, index=[
                                      "true negative", "true positive"], columns=["predicted negative", "predicted positive"])

  sn.heatmap(confusion_matrix_test, annot=True, fmt="g", cmap="viridis")

# %%

pred = model(model.x_test)[:, 1]

res=pd.DataFrame({"event_id":event_id_test,"pred":pred.detach().cpu().numpy(),
"label":test_label[:,1].cpu().numpy()})

selected_jets_idx = res.groupby(level=0)["pred"].idxmax()

efficiency=res["label"][selected_jets_idx].sum()/len(selected_jets_idx)
print(f"Efficiency: {efficiency*100:.2f}%")

#%%
fom=(df["T_mass"]-175).abs()/df["Jet_btag"]
idxmin=fom.groupby(level=0).idxmin()

fom_efficiency=df["label"][idxmin].sum()/111341


#%%
# wandb.finish()
