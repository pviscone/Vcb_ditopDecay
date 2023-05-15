#%%
import sys
sys.path.append("./JPAmodel/")
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import JPAmodel.JPANet as JPA
import JPAmodel.significance as significance
JPANet = JPA.JPANet


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


signal=torch.load("../../../root_files/signal_background/Electron/test_Electron_dataset.pt")
signal.mu_data=signal.mu_data[signal.label.squeeze()==1]
signal.nu_data=signal.nu_data[signal.label.squeeze()==1]
signal.jet_data=signal.jet_data[signal.label.squeeze()==1]
signal.label=signal.label[signal.label.squeeze()==1]

bkg=torch.load("../../../root_files/signal_background/Electron/test_Electron_dataset.pt")
bkg.mu_data=bkg.mu_data[bkg.label.squeeze()==0]
bkg.nu_data=bkg.nu_data[bkg.label.squeeze()==0]
bkg.jet_data=bkg.jet_data[bkg.label.squeeze()==0]
bkg.label=bkg.label[bkg.label.squeeze()==0]

#powheg=torch.load("../../../root_files/signal_background/Electron/powheg_Electron_dataset.pt")


#%%
mu_feat=3
nu_feat=4
jet_feat=8


model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               )
#model=torch.compile(model)
state_dict=torch.load("./state_dict.pt")
state_dict.pop("loss_fn.weight")
model.load_state_dict(state_dict)
model = model.to(device)



signal_score=torch.exp(model.predict(signal,bunch=11)[:,-1])
bkg_score=torch.exp(model.predict(bkg,bunch=100)[:,-1])
#bkg_score=torch.exp(model.predict(powheg_dataset,bunch=150)[:,-1])


#%%
importlib.reload(significance)
func=lambda x: np.arctanh(x)
significance.significance_plot(func(signal_score.detach().to(cpu).numpy()),
                            func(bkg_score.detach().to(cpu).numpy()),
                            bins=np.linspace(0,7,50),
                            ylim=(1.01e-2,2e6),
                            normalize="lumi",
                            ratio_log=True,
                            log=True,)
# %%
