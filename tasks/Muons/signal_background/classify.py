
#%%
import importlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import mplhep
sys.path.append("./JPAmodel/")
import JPAmodel.JPANet as JPA

JPANet = JPA.JPANet


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

jets_per_event = 7

#df = pd.read_pickle("./event_df.pkl", compression="bz2")
df=pd.read_pickle("./event_df.pkl")

#%%


mu_df=df.filter(regex="Muon.*(pt|eta|phi)")
nu_df=df.filter(regex="MET.*(pt|eta|phi)")
jet_df=df.filter(regex="Jet.*(pt|eta|phi|btagDeepFlavCvB|btagDeepFlavCvL|Tmass)")
#label=np.expand_dims(df["bHad_label"].astype(int).to_numpy(), axis=1)
label=df.filter(regex="(label).*").astype(int).to_numpy()

#%%

mu_data=mu_df.to_numpy()
nu_data=nu_df.to_numpy()
jet_data=jet_df.to_numpy()


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

model = JPANet(mu_data=mu_data, nu_data=nu_data, jet_data=jet_data, label=label,
               test_size=0.15, weight=None,
               mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 30, 30],
               attention_arch=[30, 30],
               event_arch=[mu_feat+nu_feat, 30, 30],
               prefinal_arch=None,
               final_attention=True,
               final_arch=[30,30*(jets_per_event+1),128,64,32],
               batch_size=5000, n_heads=2, dropout=0.1,
               optim={"lr": 0.0002, "weight_decay": 0.00, },
               early_stopping=None, shuffle=False,
               jet_mean=jet_mean, jet_std=jet_std,
               )

model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------
#model=torch.compile(model)
model.train_loop(epochs=50,show_each=12)


#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph()


# %%
signal_mask=model.y_test.squeeze()==1
bkg_mask=model.y_test.squeeze()==0
signal_score=torch.exp(model(model.mu_test[signal_mask],model.nu_test[signal_mask],model.jet_test[signal_mask])[:,-1])
bkg_score=torch.exp(model(model.mu_test[bkg_mask],model.nu_test[bkg_mask],model.jet_test[bkg_mask])[:,-1])
signal_score=signal_score.detach().to(cpu).numpy()
bkg_score=bkg_score.detach().to(cpu).numpy()
mplhep.style.use("CMS")
plt.rc('axes', axisbelow=True)


binned_signal_score=plt.hist(signal_score,bins=50,range=(0,1),
                             color="dodgerblue",edgecolor="blue",
                             histtype="stepfilled",alpha=0.8,linewidth=2,
                             #weights=np.ones_like(signal_score)*138e3*832*8.4e-4/(len(signal_score)/3),
                             label="Signal")[0]
binned_bkg_score=plt.hist(bkg_score,bins=50,range=(0,1),
                          color="red",edgecolor="red",histtype="step",
                          linewidth=2,label="Background",hatch="//",
                          #weights=np.ones_like(bkg_score)*138e3*832*(1-8.4e-4)/(len(bkg_score)/3),
         )[0]
#plt.yscale("log")
plt.legend()
plt.grid(linestyle=":")
plt.ylabel("Normalized events")
plt.xlabel("NN score")
plt.ylim(1e1,1200)
mplhep.cms.text("Private Work")
mplhep.cms.lumitext("$138 fb^{-1}$ $(13 TeV)$")
#%%
fom=binned_signal_score**2/(binned_bkg_score)
Q=(np.sum(fom[~np.isnan(fom)]))
print("Q=",Q)