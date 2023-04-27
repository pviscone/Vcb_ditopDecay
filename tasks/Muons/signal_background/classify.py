
#%%
import importlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import mplhep
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
sys.path.append("./JPAmodel/")
from JPAmodel.dataset import EventsDataset
import JPAmodel.JPANet as JPA
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


JPANet = JPA.JPANet

torch.backends.cudnn.benchmark = True
#%%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

jets_per_event = 7

#df = pd.read_pickle("./event_df.pkl", compression="bz2")
#df=pd.read_pickle("./event_df.pkl")
df=pd.read_pickle("../../../root_files/Muons/BigMuons_SB_df.pkl")

#%%


mu_data=df.filter(regex="Muon.*(pt|eta|phi)").to_numpy()
nu_data=df.filter(regex="MET.*(pt|eta|phi)").to_numpy()
jet_data=df.filter(regex="Jet.*(pt|eta|phi|btagDeepFlavCvB|btagDeepFlavCvL|Tmass)").to_numpy()
label=df.filter(regex="(label).*").astype(int).to_numpy()



mu_data=np.reshape(mu_data, (mu_data.shape[0],1, mu_data.shape[1]))
nu_data=np.reshape(nu_data, (nu_data.shape[0],1, nu_data.shape[1]))
jet_data=np.reshape(jet_data,
                    (jet_data.shape[0],
                     jets_per_event,
                     jet_data.shape[1]//jets_per_event))

mu_data = torch.tensor(mu_data, dtype=torch.float32)
nu_data = torch.tensor(nu_data, dtype=torch.float32)
jet_data = torch.tensor(jet_data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.long)


test_size=0.15
mu_train, mu_test, nu_train, nu_test,\
jet_train,jet_test,y_train,y_test=train_test_split(mu_data,nu_data,jet_data,label,test_size=test_size,shuffle=True)

train_dataset=EventsDataset(mu_train,nu_train,jet_train,y_train)
test_dataset=EventsDataset(mu_test,nu_test,jet_test,y_test)
#test_dataset.to(device)
#train_dataset.to(device)




# %%


importlib.reload(JPA)
JPANet = JPA.JPANet

mu_feat=mu_data.shape[2]
nu_feat=nu_data.shape[2]
jet_feat=jet_data.shape[2]

model = JPANet(weight=None,
               mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               attention_arch=[128, 128],
               event_arch=[mu_feat+nu_feat, 128, 128],
               prefinal_arch=None,
               final_attention=True,
               final_arch=[128,128*(jets_per_event+1),512,256,128,64,32],
               n_heads=2, dropout=0.1,
               optim={"lr": 0.0005, "weight_decay": 0.00, },
               early_stopping=None,
               )
#model=torch.compile(model)
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------
#model=torch.compile(model)
model.train_loop(train_dataset,test_dataset,epochs=50,show_each=5,train_bunch=120,test_bunch=9,batch_size=20000)


#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph(test_dataset)


# %%
signal_mask=test_dataset.label.squeeze()==1
bkg_mask=test_dataset.label.squeeze()==0
score=torch.exp(model.predict(test_dataset,bunch=10)[:,-1])
signal_score=score[signal_mask]
bkg_score=score[bkg_mask]
signal_score=signal_score.detach().to(cpu).numpy()
bkg_score=bkg_score.detach().to(cpu).numpy()
mplhep.style.use("CMS")
plt.rc('axes', axisbelow=True)


bins=40
semileptonic_weight=(138e3    #Lumi
                    *832      #Cross section
                    *0.44 #Semileptonic BR
                    *0.33     #Muon fraction
                    )
signal_weight=np.ones_like(signal_score)*semileptonic_weight*0.518*8.4e-4/(len(signal_score))
bkg_weight=np.ones_like(bkg_score)*semileptonic_weight*0.5*(1-8.4e-4)/(len(bkg_score))
binned_signal_score=plt.hist(np.arctanh(signal_score),bins=bins,range=(0,4.5),
                             color="dodgerblue",edgecolor="blue",
                             histtype="stepfilled",alpha=0.8,linewidth=2,
                             weights=signal_weight,
                             label="Signal")[0]
binned_bkg_score=plt.hist(np.arctanh(bkg_score),bins=bins,range=(0,4.5),
                          color="red",edgecolor="red",histtype="step",
                          linewidth=2,label="Background",hatch="//",
                          weights=bkg_weight,
         )[0]


signal_hist,bin_edges=np.histogram(np.arctanh(signal_score),bins=bins,range=(0,4.5))
bkg_hist,_=np.histogram(np.arctanh(bkg_score),bins=bins,range=(0,4.5))
bin_centers=(bin_edges[1:]+bin_edges[:-1])/2
plt.errorbar(bin_centers,
             binned_signal_score,
             np.sqrt(signal_hist)*binned_signal_score/signal_hist,
             fmt=",",color="black")
plt.errorbar(bin_centers,
             binned_bkg_score,
             np.sqrt(bkg_hist)*binned_bkg_score/bkg_hist,fmt=",",color="black")


plt.yscale("log")
plt.legend()
plt.grid(linestyle=":")
plt.ylabel("Normalized events")
plt.xlabel("NN score")
plt.ylim(1e-1,1e7)
mplhep.cms.text("Private Work")
mplhep.cms.lumitext("$138 fb^{-1}$ $(13 TeV)$")
#%%
bkg_not_zero=binned_bkg_score!=0
fom=binned_signal_score[bkg_not_zero]**2/(binned_bkg_score[bkg_not_zero])
Q=(np.sum(fom[~np.isnan(fom)]))
print("Q=",Q)
print(f"Error: {1/np.sqrt(Q)}")
# %%
