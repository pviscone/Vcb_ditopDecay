
#%%
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


from MLP_model import MLP

#%%
#!------------------Data loading-------------------
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("../BigMuons_Jet_features.pkl", compression="bz2")

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


#%%

train_data,test_data, train_label, test_label=train_test_split(data_df, label,test_size=test_size,shuffle=False)

train_data = torch.tensor(train_data.values, device=device)
test_data = torch.tensor(test_data.values, device=device)
train_label = torch.tensor(train_label, device=device, dtype=torch.float32)
test_label = torch.tensor(test_label, device=device, dtype=torch.float32)

assert train_data.dim() == test_data.dim()

if train_data.dim() == 1:
    train_data = train_data.unsqueeze(dim=1)
    test_data = test_data.unsqueeze(dim=1)


#!---------------------Model definition---------------------
model = MLP(x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label,
            event_id_test=event_id_test,
            hidden_arch=[80,80,80,80], batch_size=20000,
            optim={"lr": 0.003, "weight_decay": 0.000,},
            early_stopping=None
            )
model = model.to(device)

#!---------------------Training---------------------
model.train_loop(epochs=100)

#!---------------------Plot loss---------------------
model.loss_plot()

#!------------------Efficiency on events-------------------
efficiency = model.evaluate_on_events()
print(f"Efficiency on events: {efficiency}")