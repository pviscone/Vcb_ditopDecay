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

#%%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


signal=torch.load("../../../root_files/signal_background/Muons/NN/test_Muons.pt")
signal.mask(signal.data["label"]==1)


powheg=torch.load("../../../root_files/signal_background/Muons/NN/TTSemilept_MuonCuts.pt")

diLept=torch.load("../../../root_files/signal_background/Muons/NN/TTdiLept_MuonCuts.pt")
diHad=torch.load("../../../root_files/signal_background/Muons/NN/TTdiHad_MuonCuts.pt")
WJets=torch.load("../../../root_files/signal_background/Muons/NN/WJets_MuonCuts.pt")

#%%
mu_feat=3
nu_feat=3
jet_feat=6

model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               masses_arch=[36,128,128,36],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               )
model=torch.compile(model)
state_dict=torch.load("./state_dict60.pt")
state_dict.pop("_orig_mod.loss_fn.weight")
model.load_state_dict(state_dict)
model = model.to(device)


model.eval()
with torch.inference_mode():
    signal_score=torch.exp(model.predict(signal,bunch=20)[:,-1]).detach().to(cpu).numpy()
    bkg_score=torch.exp(model.predict(powheg,bunch=200)[:,-1]).detach().to(cpu).numpy()
    diLept_score=torch.exp(model.predict(diLept,bunch=20)[:,-1]).detach().to(cpu).numpy()
    diHad_score=torch.exp(model.predict(diHad,bunch=20)[:,-1]).detach().to(cpu).numpy()
    WJets_score=torch.exp(model.predict(WJets,bunch=20)[:,-1]).detach().to(cpu).numpy()




#%%
lumi=138e3
ttbar_1lept=lumi*832*0.44  #all lepton
ttbar_2had=lumi*832*0.45*0.0032  #all quark types
ttbar_2lept=lumi*832*0.11*0.2365 #all lepton types
wjets=lumi*59100*0.108*3*0.0003 #all lepton types

tau_mask=(torch.abs(powheg.data["LeptLabel"])==15).squeeze()
not_tau_mask=(torch.abs(powheg.data["LeptLabel"])!=15).squeeze()

additional_c=(np.abs(powheg.data["AdditionalPartons"])==4).squeeze()
additional_b=(np.abs(powheg.data["AdditionalPartons"])==5).squeeze()
had_decay=np.abs(powheg.data["HadDecay"])
charmed=np.bitwise_or(had_decay[:,0]==4,had_decay[:,1]==4).bool()
up=np.bitwise_or(had_decay[:,0]==2,had_decay[:,1]==2).bool()

func = lambda x: np.arctanh(x)
sig_score=func(signal_score)
tt_tau=func(bkg_score[tau_mask])
ttc=func(bkg_score[np.bitwise_and(additional_c,not_tau_mask).bool()])
ttb=func(bkg_score[np.bitwise_and(additional_b,not_tau_mask).bool()])
tt_charmed=func(bkg_score[np.bitwise_and(charmed,not_tau_mask).bool()])
tt_up=func(bkg_score[np.bitwise_and(up,not_tau_mask).bool()])

TTdiLept=func(diLept_score)
TTdiHad=func(diHad_score)
WJets_bkg=func(WJets_score)


#%%

importlib.reload(significance)
hist_dict = {
            "signal":{
                "data":sig_score,
                "color":"red",
                "weight":ttbar_1lept*0.518*0.33*8.4e-4,
                "histtype":"errorbar",
                "stack":False,},
            "$t\\bar{t}+b$":{
                "data":ttb,
                "color":"cornflowerblue",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(additional_b)/len(additional_b),
                "stack":True,},
            "$t\\bar{t}+c$":{
                "data":ttc,
                "color":"lightsteelblue",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(additional_c)/len(additional_c),
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} uql \\nu$":{
                "data":tt_up,
                "color":"plum",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(up)/len(up),
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} cql \\nu$":{
                "data":tt_charmed,
                "color":"lightcoral",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(charmed)/len(charmed),
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} q \\bar{q} \\tau \\nu_{\\tau}$":{
                "data":tt_tau,
                "color":"cadetblue",
                "weight":ttbar_1lept*(1-8.4e-4)*0.178*torch.sum(tau_mask)/len(tau_mask),
                "stack":True,},
                        "$Wj \\to l \\nu$":{
            "data":WJets_bkg,
                "color":"limegreen",
                "weight":wjets,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} q \\bar{q} q \\bar{q}$":{
                "data":TTdiHad,
                "color":"orange",
                "weight":ttbar_2had,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} l \\nu l \\nu$":{
                "data":TTdiLept,
                "color":"gold",
                "weight":ttbar_2lept,
                "stack":True,},
            
        }

"""
            "$Wj \\to l \\nu$":{
                "data":WJets_bkg,
                "color":"limegreen",
                "weight":wjets,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} q \\bar{q} q \\bar{q}$":{
                "data":TTdiHad,
                "color":"orange",
                "weight":ttbar_2had,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} l \\nu l \\nu$":{
                "data":TTdiLept,
                "color":"gold",
                "weight":ttbar_2lept,
                "stack":True,},
            """


ax1,ax2=significance.make_hist(hist_dict,xlim=(0,7),bins=60,log=True)

#%%


def get(obj):
    mean=powheg.stats_dict["jet_mean"][0].detach().numpy()
    std=powheg.stats_dict["jet_std"][0].detach().numpy()
    pt=obj.jet_data[:,3,0].detach().numpy()
    return 100*std*np.tanh(pt)+mean


importlib.reload(significance)
hist_dict = {
            "signal (renormalized)":{
                "data": get(signal),
                "color":"red",
                "weight":ttbar_2lept+ttbar_2had+wjets+ttbar_1lept*0.5*(1-8.4e-4)/100,
                "histtype":"step",
                "stack":False,},
            "$t\\bar{t}j \\to b\\bar{b} q \\bar{q} l \\nu /100$":{
                "data":get(powheg),
                "color":"cornflowerblue",
                "weight":ttbar_1lept*0.5*(1-8.4e-4)/100,
                "stack":True,},
            "$Wj \\to l \\nu$":{
                "data":get(WJets),
                "color":"limegreen",
                "weight":wjets,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} q \\bar{q} q \\bar{q}$":{
                "data":get(diHad),
                "color":"orange",
                "weight":ttbar_2had,
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} l \\nu l \\nu$":{
                "data":get(diLept),
                "color":"gold",
                "weight":ttbar_2lept,
                "stack":True,},
            
            
        }

ax1,ax2=significance.make_hist(hist_dict,xlim=(20,450),bins=80,log=True,significance=False,density=False,ylim=(1e-3,3e6))
plt.ylim(3e-1,2e1)
ax2.set_xlabel("$p_t$ [GeV]")
ax2.set_ylabel("Sig./Bkg.")
# %%
# %%
hist_kwargs = { "bins":50, "density":True, "log":True, "range":(0,5),"histtype":"step", "linewidth":2}
plt.hist(sig_score,**hist_kwargs,color="red",label="signal")
plt.hist(tt_tau,**hist_kwargs,color="cadetblue",label="$t\\bar{t} j \\to b\\bar{b} q \\bar{q} \\tau \\nu_{\\tau}$")
plt.hist(ttc,**hist_kwargs,color="lightsteelblue",label="$t\\bar{t}+c$")
plt.hist(ttb,**hist_kwargs,color="cornflowerblue",label="$t\\bar{t}+b$")
plt.hist(tt_charmed,**hist_kwargs,color="lightcoral",label="$t\\bar{t} \\to b\\bar{b} cql \\nu$")
plt.hist(tt_up,**hist_kwargs,color="plum",label="$t\\bar{t} \\to b\\bar{b} uql \\nu$")
plt.hist(WJets_bkg,**hist_kwargs,color="limegreen",label="$Wj \\to l \\nu$")
plt.hist(TTdiHad,**hist_kwargs,color="orange",label="$t\\bar{t} \\to b\\bar{b} q \\bar{q} q \\bar{q}$")
plt.hist(TTdiLept,**hist_kwargs,color="gold",label="$t\\bar{t} \\to b\\bar{b} l \\nu l \\nu$")
plt.legend(fontsize=18)
plt.ylim(1e-6,1e3)


#%%
import corner
import torch
import numpy as np
from JPAmodel.torch_dataset import EventsDataset

#%%
powheg=torch.load("../../../root_files/signal_background/Muons/NN/TTSemilept_MuonCuts.pt")
powheg.slice(0,100000)

#%%
tt_charmed_nu,tt_charmed_mu,tt_charmed_jet,_=powheg[np.bitwise_and(charmed,not_tau_mask).bool()]
tt_up_nu,tt_up_mu,tt_up_jet,_=powheg[np.bitwise_and(up,not_tau_mask).bool()]

tt_charmed_jet_out=np.repeat(tt_charmed,7)


tt_charmed_jet=torch.flatten(tt_charmed_jet,end_dim=1)
tt_up_jet=torch.flatten(tt_up_jet,end_dim=1)


# %%
fig=corner.corner(tt_charmed_jet,range=[[0,200]]+[[-3.14,3.14]]+[[-6,6]]+[[0,1]]*3,
              hist_kwargs={"ls": "--"},
              bins=50,
              contour_kwargs={"linestyles": "--"},
              color="tab:blue",)

corner.corner(tt_up_jet,range=[[0,200]]+[[-3.14,3.14]]+[[-6,6]]+[[0,1]]*3,
              hist_kwargs={"ls": "--"},
              bins=50,
              contour_kwargs={"linestyles": "--"},
              color="tab:orange",
              fig=fig)
# %%
