#%%
import sys
sys.path.append("./JPAmodel/")
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import JPAmodel.JPANet as JPA
import JPAmodel.significance as significance
import corner
JPANet = JPA.JPANet

#%%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

path="/scratchnvme/pviscone/Preselection_Skim/NN/predict/torch/"
signal=torch.load(path+"signal_predict_MuonCuts.pt")
signal_Electrons=torch.load(path+"signal_Electrons_predict_MuonCuts.pt")
signal_Taus=torch.load(path+"signal_Taus_predict_MuonCuts.pt")
semiLept=torch.load(path+"TTSemiLept_predict_MuonCuts.pt")
diLept=torch.load(path+"TTdiLept_predict_MuonCuts.pt")
diHad=torch.load(path+"TTdiHad_predict_MuonCuts.pt")
WJets=torch.load(path+"WJets_predict_MuonCuts.pt")

#%%
mu_feat=3
nu_feat=3
jet_feat=6

model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               masses_arch=[36,128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               secondLept_arch=[3,128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               n_jet=7,
               )
#model=torch.compile(model)
state_dict=torch.load("./state_dict_110_final.pt",map_location=torch.device(device))
state_dict.pop("loss_fn.weight")
model.load_state_dict(state_dict)
model = model.to(device)


model.eval()
with torch.inference_mode():
    signal_score=torch.exp(model.predict(signal,bunch=1)[:,-1]).detach().to(cpu).numpy()
    signal_Electrons_score=torch.exp(model.predict(signal_Electrons,bunch=1)[:,-1]).detach().to(cpu).numpy()
    signal_Taus_score=torch.exp(model.predict(signal_Taus,bunch=1)[:,-1]).detach().to(cpu).numpy()
    bkg_score=torch.exp(model.predict(semiLept,bunch=1)[:,-1]).detach().to(cpu).numpy()
    diLept_score=torch.exp(model.predict(diLept,bunch=1)[:,-1]).detach().to(cpu).numpy()
    diHad_score=torch.exp(model.predict(diHad,bunch=1)[:,-1]).detach().to(cpu).numpy()
    WJets_score=torch.exp(model.predict(WJets,bunch=1)[:,-1]).detach().to(cpu).numpy()

#%%
save_path="/scratchnvme/pviscone/Preselection_Skim/NN/scores/"
torch.save(signal_score,save_path+"/signal_score_Muons.pt")
torch.save(signal_Electrons_score,save_path+"/signal_Electrons_score_Muons.pt")
torch.save(signal_Taus_score,save_path+"/signal_Taus_score_Muons.pt")
torch.save(bkg_score,save_path+"/bkg_score_Muons.pt")
torch.save(diLept_score,save_path+"/diLept_score_Muons.pt")
torch.save(diHad_score,save_path+"/diHad_score_Muons.pt")
torch.save(WJets_score,save_path+"/WJets_score_Muons.pt")


#%%
lumi=138e3
ttbar_1lept=lumi*832*0.44  #all lepton
ttbar_2had=lumi*832*0.45*0.0032  #all quark types
ttbar_2lept=lumi*832*0.11*0.2365 #all lepton types
wjets=lumi*59100*0.108*3*0.0003 #all lepton types

tau_mask=(torch.abs(semiLept.data["LeptLabel"])==15).squeeze()
not_tau_mask=(torch.abs(semiLept.data["LeptLabel"])!=15).squeeze()

additional_c=(np.abs(semiLept.data["AdditionalPartons"])==4).squeeze()
additional_b=(np.abs(semiLept.data["AdditionalPartons"])==5).squeeze()
had_decay=np.abs(semiLept.data["HadDecay"])
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
signal_Electrons_score=func(signal_Electrons_score)
signal_Taus_score=func(signal_Taus_score)

#%%

importlib.reload(significance)
hist_dict = {
            "signal Muons":{
                "data":sig_score,
                "color":"red",
                "weight":ttbar_1lept*0.518*0.33*8.4e-4,
                "histtype":"step",
                "stack":False,},
            "$t\\bar{t}+b$":{
                "data":ttb,
                "color":"palegreen",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(additional_b)/len(additional_b),
                "stack":True,},
            "$t\\bar{t}+c$":{
                "data":ttc,
                "color":"cornflowerblue",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(additional_c)/len(additional_c),
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} uql \\nu$":{
                "data":tt_up,
                "color":"pink",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(up)/len(up),
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} cql \\nu$":{
                "data":tt_charmed,
                "color":"lavender",
                "weight":ttbar_1lept*0.178*(1-8.4e-4)*torch.sum(charmed)/len(charmed),
                "stack":True,},
            "$Wj \\to l \\nu$":{
                "data":WJets_bkg,
                "color":"limegreen",
                "weight":wjets,
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} q \\bar{q} q \\bar{q}$":{
                "data":TTdiHad,
                "color":"gold",
                "weight":ttbar_2had,
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} l \\nu l \\nu$":{
                "data":TTdiLept,
                "color":"orange",
                "weight":ttbar_2lept,
                "stack":True,},
        }



ax1,ax2=significance.make_hist(hist_dict,xlim=(0,6),bins=60,log=True,ylim=(1e-1,1e7))
ax2.set_ylim(1e-4,2e1)

# %%
hist_kwargs = { "bins":50, "density":True, "log":True, "range":(0,5),"histtype":"step", "linewidth":2}
plt.hist(sig_score,**hist_kwargs,color="red",label="signal")
plt.hist(signal_Electrons_score,**hist_kwargs,color="darkorange",label="$t\\bar{t} j \\to b\\bar{b} cb e \\nu$")
plt.hist(signal_Taus_score,**hist_kwargs,color="chocolate",label="$t\\bar{t} j \\to b\\bar{b} cb \\tau \\nu$")
plt.hist(tt_tau,**hist_kwargs,color="cadetblue",label="$t\\bar{t} j \\to b\\bar{b} q \\bar{q} \\tau \\nu_{\\tau}$")
plt.hist(ttc,**hist_kwargs,color="lightsteelblue",label="$t\\bar{t}+c$")
plt.hist(ttb,**hist_kwargs,color="cornflowerblue",label="$t\\bar{t}+b$")
plt.hist(tt_charmed,**hist_kwargs,color="lightcoral",label="$t\\bar{t} \\to b\\bar{b} cql \\nu$")
plt.hist(tt_up,**hist_kwargs,color="plum",label="$t\\bar{t} \\to b\\bar{b} uql \\nu$")
plt.hist(WJets_bkg,**hist_kwargs,color="limegreen",label="$Wj \\to l \\nu$")
plt.hist(TTdiHad,**hist_kwargs,color="orange",label="$t\\bar{t} \\to b\\bar{b} q \\bar{q} q \\bar{q}$")
plt.hist(TTdiLept,**hist_kwargs,color="gold",label="$t\\bar{t} \\to b\\bar{b} l \\nu l \\nu$")
plt.legend(fontsize=16)
plt.ylim(1e-6,1e3)




# %%

"""
fig=corner.corner(tt_charmed.data["Jet"][:100000],range=[[0,200]]+[[-3.14,3.14]]+[[-6,6]]+[[0,1]]*3,
              hist_kwargs={"ls": "--"},
              bins=50,
              contour_kwargs={"linestyles": "--"},
              color="tab:blue",)

corner.corner(tt_up.data["Jet"][:100000],range=[[0,200]]+[[-3.14,3.14]]+[[-6,6]]+[[0,1]]*3,
              hist_kwargs={"ls": "--"},
              bins=50,
              contour_kwargs={"linestyles": "--"},
              color="tab:orange",
              fig=fig)
"""
# %%
