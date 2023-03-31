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

jets_per_event = 6

df = pd.read_pickle("./event_df.pkl", compression="bz2")
#df=pd.read_pickle("./BigMuons_event_df.pkl")

#%%
weights=[0.]


label=np.expand_dims(df["bLept_label"].astype(int).to_numpy(), axis=1)

weights = np.histogram(label.squeeze(), density=True, range=(0, jets_per_event), bins=jets_per_event)[0]
weights=torch.tensor(weights,dtype=torch.float32, device=device)

weights=1/weights
weights=weights/weights.sum()


#df=df.loc[:, df.columns != "label"]
df=df.drop(columns=["Muon_mass", "Neutrino_mass"])


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



temp_jet_df=pd.DataFrame(torch.flatten(jet_data,end_dim=1).to(cpu).numpy())
jet_mean=torch.tensor(temp_jet_df[temp_jet_df!=0].mean().to_numpy(),dtype=torch.float32, device=device)
jet_std=torch.tensor(temp_jet_df[temp_jet_df!=0].std().to_numpy(),dtype=torch.float32, device=device)

del temp_jet_df

# %%
importlib.reload(JPA)
JPANet = JPA.JPANet

mu_feat=mu_data.shape[2]
nu_feat=nu_data.shape[2]
jet_feat=jet_data.shape[2]

model =JPANet(mu_data=mu_data,nu_data=nu_data,jet_data=jet_data,label=label,test_size=0.15,weight=None,
            mu_arch=None,nu_arch=None,jet_arch=[jet_feat,50,50],
            attention_arch=None,
            event_arch=[mu_feat+nu_feat,50,50],
            prefinal_arch=None,
            final_attention=True,
            final_arch=[50,50],
            batch_size=20000, n_heads=2, dropout=0.1,
            optim={"lr": 0.008, "weight_decay": 0.00, },
            early_stopping=None,shuffle=False,
            jet_mean=jet_mean,jet_std=jet_std,
            )
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------
torch.compile(model)
model.train_loop(epochs=500,show_each=12)

#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph()


# %%
