#%%
import sys
sys.path.append("./JPAmodel/")


import importlib
import pandas as pd
import torch
import numpy as np
import JPAmodel.JPANet as JPA

JPANet = JPA.JPANet


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

jets_per_event = 7

mu_data=df.loc[:, df.columns.str.contains("Muon")].to_numpy()
nu_data=df.loc[:, df.columns.str.contains("Neutrino")].to_numpy()
jet_data=df.loc[:, df.columns.str.contains("Jet")].to_numpy()

mu_data=np.reshape(mu_data, (mu_data.shape[0],1, mu_data.shape[1]))
nu_data=np.reshape(nu_data, (nu_data.shape[0],1, nu_data.shape[1]))
jet_data=np.reshape(jet_data, (jet_data.shape[0],jets_per_event, jet_data.shape[1]//jets_per_event))


mu_data = torch.tensor(mu_data, dtype=torch.float32, device=device)
nu_data = torch.tensor(nu_data, dtype=torch.float32, device=device)
jet_data = torch.tensor(jet_data, dtype=torch.float32, device=device)
label = torch.tensor(label, dtype=torch.long, device=device)



#%%

importlib.reload(JPA)
JPANet = JPA.JPANet



model =JPANet(mu_data=mu_data,nu_data=nu_data,jet_data=jet_data,label=label,test_size=0.15,
            mu_arch=None,nu_arch=None,jet_arch=None,
            event_arch=[50,50],attention_arch=[50,50],final_arch=[50,50],
            batch_size=20000, n_heads=1, dropout=0.1,
            optim={"lr": 0.001, "weight_decay": 0.00, },
            early_stopping=None,shuffle=False,
            )
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")
model.graph()
#!---------------------Training---------------------

model.train_loop(epochs=500,show_each=20)

#!---------------------Plot loss---------------------
model.loss_plot()



# %%
