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


signal=torch.load("../../../root_files/signal_background/Muons/test_Muon_dataset.pt")
signal.mu_data=signal.mu_data[signal.label.squeeze()==1]
signal.nu_data=signal.nu_data[signal.label.squeeze()==1]
signal.jet_data=signal.jet_data[signal.label.squeeze()==1]
signal.label=signal.label[signal.label.squeeze()==1]

""" bkg=torch.load("../../../root_files/signal_background/Muons/test_Muon_dataset.pt")
bkg.mu_data=bkg.mu_data[bkg.label.squeeze()==0]
bkg.nu_data=bkg.nu_data[bkg.label.squeeze()==0]
bkg.jet_data=bkg.jet_data[bkg.label.squeeze()==0]
bkg.label=bkg.label[bkg.label.squeeze()==0]
 """
powheg=torch.load("../../../root_files/signal_background/Muons/powheg_Muon_dataset.pt")


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



signal_score=torch.exp(model.predict(signal,bunch=11)[:,-1]).detach().to(cpu).numpy()
#bkg_score=torch.exp(model.predict(bkg,bunch=100)[:,-1])
bkg_score=torch.exp(model.predict(powheg,bunch=150)[:,-1]).detach().to(cpu).numpy()



#%%
importlib.reload(significance)
func=lambda x: np.arctanh(x)
significance.significance_plot(func(signal_score),
                            func(bkg_score),
                            bins=np.linspace(0,7,50),
                            ylim=(1e-3,1e7),
                            normalize="lumi",
                            ratio_log=True,
                            log=True,)
# %%
importlib.reload(significance)

lumi=138e3
ttbar_1lept=lumi*832*0.44*0.33



additional_c=np.abs(powheg.additional_parton)==4
additional_b=np.abs(powheg.additional_parton)==5
had_decay=np.abs(powheg.had_decay)
charmed=np.bitwise_or(had_decay[:,0]==4,had_decay[:,1]==4).bool()
up=np.bitwise_or(had_decay[:,0]==2,had_decay[:,1]==2).bool()


func = lambda x: np.arctanh(x)
hist_dict={ "signal":{"data":func(signal_score),
                     "color":"crimson",
                     "weight":ttbar_1lept*0.518*8.4e-4,
                     "stack":False,},
            "additional c":{"data":func(bkg_score[additional_c]),
                          "color":"dodgerblue",
                          "weight":ttbar_1lept*0.5*(1-8.4e-4)*torch.sum(additional_c)/len(additional_c),
                          "stack":True,},
            "additional b":{"data":func(bkg_score[additional_b]),
                            "color":"orchid",
                            "weight":ttbar_1lept*0.5*(1-8.4e-4)*torch.sum(additional_b)/len(additional_b),
                            "stack":True,},
            "$b\\bar{b} uql \\nu$":{"data":func(bkg_score[up]),
                                    "color":"lime",
                                    "weight":ttbar_1lept*0.5*(1-8.4e-4)*torch.sum(up)/len(up),
                                    "stack":True,},
            "$b\\bar{b} cql \\nu$":{"data":func(bkg_score[charmed]),
                                    "color":"orange",
                                    "weight":ttbar_1lept*0.5*(1-8.4e-4)*torch.sum(charmed)/len(charmed),
                                    "stack":True,},
}

ax1,ax2=significance.make_hist(hist_dict,xlim=(0,7),bins=60,log=True)
#%%