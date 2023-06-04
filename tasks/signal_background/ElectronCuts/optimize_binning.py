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


signal=torch.load("../../../root_files/signal_background/Electrons/test_Electron_dataset.pt")
signal.mu_data=signal.mu_data[signal.label.squeeze()==1]
signal.nu_data=signal.nu_data[signal.label.squeeze()==1]
signal.jet_data=signal.jet_data[signal.label.squeeze()==1]
signal.label=signal.label[signal.label.squeeze()==1]

""" bkg=torch.load("../../../root_files/signal_background/Electrons/test_Electron_dataset.pt")
bkg.mu_data=bkg.mu_data[bkg.label.squeeze()==0]
bkg.nu_data=bkg.nu_data[bkg.label.squeeze()==0]
bkg.jet_data=bkg.jet_data[bkg.label.squeeze()==0]
bkg.label=bkg.label[bkg.label.squeeze()==0] """

powheg=torch.load("../../../root_files/signal_background/Electrons/powheg_Electron_dataset.pt")

diLept=torch.load("../../../root_files/signal_background/Electrons/TTdiLept_powheg_Electron_dataset.pt")
diHad=torch.load("../../../root_files/signal_background/Electrons/TTdiHad_powheg_Electron_dataset.pt")
WJets=torch.load("../../../root_files/signal_background/Electrons/WJets_Electron_dataset.pt")


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




model.eval()
with torch.inference_mode():
    signal_score=torch.exp(model.predict(signal,bunch=20)[:,-1]).detach().to(cpu).numpy()
    #bkg_score=torch.exp(model.predict(bkg,bunch=100)[:,-1])
    bkg_score=torch.exp(model.predict(powheg,bunch=200)[:,-1]).detach().to(cpu).numpy()
    diLept_score=torch.exp(model.predict(diLept,bunch=20)[:,-1]).detach().to(cpu).numpy()
    diHad_score=torch.exp(model.predict(diHad,bunch=20)[:,-1]).detach().to(cpu).numpy()
    WJets_score=torch.exp(model.predict(WJets,bunch=20)[:,-1]).detach().to(cpu).numpy()




#%%


lumi=138e3
ttbar_1lept=lumi*832*0.44*0.33  #single lepton
ttbar_2had=lumi*832*0.45*0.001  #all quark types
ttbar_2lept=lumi*832*0.11*0.149 #all lepton types
wjets=lumi*59100*0.108*3*0.0002 #all lepton types

tau_mask=(powheg.Lept_label==15).squeeze()
not_tau_mask=(powheg.Lept_label!=15).squeeze()

additional_c=np.abs(powheg.additional_parton)==4
additional_b=np.abs(powheg.additional_parton)==5
had_decay=np.abs(powheg.had_decay)
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

#ATTENTO, non hai incluso il peso dei tau
importlib.reload(significance)
hist_dict = {
            "signal":{
                "data":sig_score,
                "color":"red",
                "weight":ttbar_1lept*0.363*8.4e-4,
                "histtype":"errorbar",
                "stack":False,},
            "$t\\bar{t}+b$":{
                "data":ttb,
                "color":"cornflowerblue",
                "weight":ttbar_1lept*0.352*(1-8.4e-4)*torch.sum(additional_b)/len(additional_b),
                "stack":True,},
            "$t\\bar{t}+c$":{
                "data":ttc,
                "color":"lightsteelblue",
                "weight":ttbar_1lept*0.352*(1-8.4e-4)*torch.sum(additional_c)/len(additional_c),
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} uql \\nu$":{
                "data":tt_up,
                "color":"plum",
                "weight":ttbar_1lept*0.352*(1-8.4e-4)*torch.sum(up)/len(up),
                "stack":True,},
            "$t\\bar{t} \\to b\\bar{b} cql \\nu$":{
                "data":tt_charmed,
                "color":"lightcoral",
                "weight":ttbar_1lept*0.352*(1-8.4e-4)*torch.sum(charmed)/len(charmed),
                "stack":True,},
            "$t\\bar{t} j \\to b\\bar{b} q \\bar{q} \\tau \\nu_{\\tau}$":{
                "data":tt_tau,
                "color":"cadetblue",
                "weight":ttbar_1lept*(1-8.4e-4)*0.022,
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

ax1,ax2=significance.make_hist(hist_dict,xlim=(0,5),bins=50,log=True)


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

# %%