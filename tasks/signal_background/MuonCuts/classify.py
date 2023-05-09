
#%%
import importlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import mplhep
sys.path.append("./JPAmodel/")
from JPAmodel.dataset import build_datasets
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


#%%
train_dataset=torch.load("../../../root_files/signal_background/train_dataset.pt")
test_dataset=torch.load("../../../root_files/signal_background/test_dataset.pt")
powheg_dataset=torch.load("../../../root_files/signal_background/powheg_dataset.pt")

#test_dataset.to(device)
#train_dataset.to(device)
#powheg_dataset.to(device)


# %%

importlib.reload(JPA)
JPANet = JPA.JPANet

mu_feat=3
nu_feat=4
jet_feat=8

model = JPANet(weight=None,
               mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               optim={"lr": 0.001, "weight_decay": 0.00, },
               early_stopping=None,
               )
#model=torch.compile(model)
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------
#model=torch.compile(model)
#torch.save(model.state_dict(), "./state_dict.pt")
#model.state_dict=torch.load("./state_dict.pt")
model.train_loop(train_dataset,test_dataset,epochs=50,show_each=5,train_bunch=25,test_bunch=3,batch_size=20000)


#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph(test_dataset)


# %%
#!Put in other file

signal_mask=test_dataset.label.squeeze()==1
bkg_mask=test_dataset.label.squeeze()==0
score=torch.exp(model.predict(test_dataset,bunch=10)[:,-1])
signal_score=score[signal_mask]
bkg_score=torch.exp(model.predict(powheg_dataset,bunch=150)[:,-1])
#bkg_score=score[bkg_mask]

signal_score=signal_score.detach().to(cpu).numpy()
bkg_score=bkg_score.detach().to(cpu).numpy()
mplhep.style.use("CMS")
plt.rc('axes', axisbelow=True)

#%%
bins=20
score_range=(0,5.5)
semileptonic_weight=(138e3    #Lumi
                    *832      #Cross section
                    *0.44 #Semileptonic BR
                    *0.33     #Muon fraction
                    )
signal_weight=np.ones_like(signal_score)*semileptonic_weight*0.518*8.4e-4/(len(signal_score))
bkg_weight=np.ones_like(bkg_score)*semileptonic_weight*0.5*(1-8.4e-4)/(len(bkg_score))



binned_signal_score=plt.hist(np.arctanh(signal_score),bins=bins,range=score_range,
                             color="dodgerblue",edgecolor="blue",
                             histtype="stepfilled",alpha=0.8,linewidth=2,
                             weights=signal_weight,
                             label="Signal")[0]
binned_bkg_score=plt.hist(np.arctanh(bkg_score),bins=bins,range=score_range,
                          color="red",edgecolor="red",histtype="step",
                          linewidth=2,label="Background",hatch="//",
                          weights=bkg_weight,
         )[0]


signal_hist,bin_edges=np.histogram(np.arctanh(signal_score),bins=bins,range=score_range)
bkg_hist,_=np.histogram(np.arctanh(bkg_score),bins=bins,range=score_range)

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


zero_bkg=binned_bkg_score==0
fom=binned_signal_score**2/(binned_bkg_score+binned_signal_score)
Q=(np.sum(fom[~np.isnan(fom)]))
print("Q=",Q)
print(f"Error: {1/np.sqrt(Q)}")
# %%
