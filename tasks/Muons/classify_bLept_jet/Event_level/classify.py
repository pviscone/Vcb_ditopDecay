# %%
import sys
sys.path.append("./JPAmodel/")


import JPAmodel.JPANet as JPA
import numpy as np
import torch
import pandas as pd
import importlib


JPANet = JPA.JPANet
#%%

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

jets_per_event = 7

df = pd.read_pickle("./BigMuons_event_df.pkl", compression="bz2")


label = np.expand_dims(df["label"].astype(int).to_numpy(), axis=1)

temp_label = np.zeros((label.shape[0], jets_per_event), dtype=bool)
np.put_along_axis(temp_label, label, True, axis=1)
# temp_label=np.expand_dims(temp_label, axis=2)
temp_label = np.stack([np.bitwise_not(temp_label), temp_label], axis=2)
label = temp_label.astype(np.float32)

del temp_label


df = df.loc[:, df.columns != "label"]
df = df.drop(columns=["Muon_mass", "Neutrino_mass"])


mu_data = df.loc[:, df.columns.str.contains("Muon")].to_numpy()
nu_data = df.loc[:, df.columns.str.contains("Neutrino")].to_numpy()

jet_features = ["btagDeepFlavCvB", "pt", "phi", "eta",
           "mass","Tmass"]
jet_col=[f"Jet{i}_{column}" for i in range(jets_per_event) for column in jet_features]

jet_df = df[jet_col]


jet_data = jet_df.loc[:, jet_df.columns.str.contains("Jet")].to_numpy()

mu_data = np.reshape(mu_data, (mu_data.shape[0], 1, mu_data.shape[1]))
nu_data = np.reshape(nu_data, (nu_data.shape[0], 1, nu_data.shape[1]))
jet_data = np.reshape(
    jet_data, (jet_data.shape[0], jets_per_event, jet_data.shape[1]//jets_per_event))


mu_data = torch.tensor(mu_data, dtype=torch.float32, device=device)
nu_data = torch.tensor(nu_data, dtype=torch.float32, device=device)
jet_data = torch.tensor(jet_data, dtype=torch.float32, device=device)
label = torch.tensor(label, dtype=torch.long, device=device)


# %%

importlib.reload(JPA)
JPANet = JPA.JPANet


model = JPANet(mu_data=mu_data, nu_data=nu_data, jet_data=jet_data, label=label, test_size=0.15,
               mu_arch=None, nu_arch=None, jet_arch=[50,50],
               event_arch=[50, 50], attention_arch=[50], final_arch=[80,80,80], final_attention=,
               batch_size=30000, n_heads=1, dropout=0.15,
               optim={"lr": 0.003, "weight_decay": 0.00, },
               early_stopping=None, shuffle=False,
               )
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------

model.train_loop(epochs=500, show_each=20)

#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph()


# %%
