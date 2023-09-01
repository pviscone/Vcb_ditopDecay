# %%
import uproot
import correctionlib
import numpy as np
import pandas as pd
import awkward as ak
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import copy
import matplotlib.pyplot as plt
import mplhep

b_ceval=correctionlib.CorrectionSet.from_file("json/btagging.json")
c_ceval=correctionlib.CorrectionSet.from_file("json/ctagging.json")
btag=correctionlib_wrapper(b_ceval["deepJet_shape"])
ctag=correctionlib_wrapper(c_ceval["deepJet_shape"])


def np_and(*args):
    a=args[0]
    for arg in args[1:]:
        a=np.bitwise_and(a,arg)
    return a

def np_or(*args):
    a=args[0]
    for arg in args[1:]:
        a=np.bitwise_or(a,arg)
    return a



    

f=uproot.open("/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/predict/3A361FB1-E533-1C4D-9EA7-ACFA17F05B69.root")

arrays=f["Events"].arrays(["Jet_pt","Jet_eta","Jet_btagDeepFlavB","Jet_btagDeepFlavCvL","Jet_btagDeepFlavCvB","Jet_hadronFlavour","Jet_jetId","Jet_puId"])


arrays=arrays[np_and(
                    arrays["Jet_pt"]>20,
                    np.abs(arrays["Jet_eta"])<2.5,
                    arrays["Jet_jetId"]>0,
                    arrays["Jet_puId"]>0
                    )
              ]
n_ev=1300000
arrays=arrays[:n_ev,:7]
assert len(arrays)==n_ev
arrays=ak.pad_none(arrays,7,clip=True,axis=1)
arrays=ak.fill_none(arrays,-1)



def evaluate_btag(array,name):
    array["b_"+name]=ak.Array([[1.,1.,1.,1.,1.,1.,1.]]*n_ev)
    mask=np.asarray(ak.flatten(array["Jet_hadronFlavour"]))>-1
    if name=="central":
        pass
    elif name.split("_")[1]=="cferr1" or name.split("_")[1]=="cferr2":
        mask=np_and(mask,np.asarray(ak.flatten(array["Jet_hadronFlavour"]))==4)
    else:
        mask=np_and(mask,np.asarray(ak.flatten(array["Jet_hadronFlavour"]))!=4)

    #mask=np_and(mask,np.asarray(ak.flatten(np.abs(array["Jet_eta"])))<2.5)
    
    np.asarray(ak.flatten(array["b_"+name]))[mask]=btag(name,
                                                    np.asarray(ak.flatten(array["Jet_hadronFlavour"]))[mask],
                                                    np.asarray(ak.flatten(np.abs(array["Jet_eta"])))[mask],
                                                    np.asarray(ak.flatten(array["Jet_pt"]))[mask],
                                                    np.asarray(ak.flatten(array["Jet_btagDeepFlavB"]))[mask])
    
    return array

def evaluate_ctag(array,name):
    array["c_"+name]=ak.Array([[1.,1.,1.,1.,1.,1.,1.]]*n_ev)
    mask=np.asarray(ak.flatten(array["Jet_hadronFlavour"]))>-1
    #mask=np_and(mask,np.asarray(ak.flatten(np.abs(array["Jet_eta"])))<2.5)
    np.asarray(ak.flatten(array[f"c_{name}"]))[mask]=ctag(name,
                     np.asarray(ak.flatten(array["Jet_hadronFlavour"]))[mask],
                     np.asarray(ak.flatten(array["Jet_btagDeepFlavCvL"]))[mask],
                     np.asarray(ak.flatten(array["Jet_btagDeepFlavCvB"]))[mask])
    return array


# %%
b_sources=["hf","lf","hfstats1","hfstats2","lfstats1","lfstats2","cferr1","cferr2"]
b_systs=[]

for b_source in b_sources:
    b_systs.append("up_"+b_source)
    b_systs.append("down_"+b_source)
    
    
arrays=evaluate_btag(arrays,"central")
for b_syst in b_systs:
    arrays=evaluate_btag(arrays,b_syst)



df=ak.to_dataframe(arrays)
b_systs=["b_"+b_syst for b_syst in b_systs]
df=df[b_systs+["b_central","Jet_hadronFlavour","Jet_btagDeepFlavB"]]


c_df=df[df["Jet_hadronFlavour"]==4]
b_df=df[(df["Jet_hadronFlavour"]==5)]
l_df=df[(df["Jet_hadronFlavour"]==0)]

print(f'btag_cferr1/2 \t means: (on hadFlav==4) \n----------------------------\n{c_df.mean()[["b_up_cferr1","b_down_cferr1","b_up_cferr2","b_down_cferr2"]]}')

print("\n##############################\n")

print(f'btag !=cferr1/2\t means: (on hadFlav==5) \n----------------------------\n{b_df.mean()}')
print("\n##############################\n")
print(f'btag !=cferr1/2\t means: (on hadFlav==0) \n----------------------------\n{l_df.mean()}')
#%%
df2=df[df["Jet_hadronFlavour"]!=-1]
print(f"bTag mean on events (not normalized):\n{df2.groupby(level=0).prod().mean()}")

c_mu=df2[df2["Jet_hadronFlavour"]==4].mean()
b_mu=df2[df2["Jet_hadronFlavour"]==5].mean()
l_mu=df2[df2["Jet_hadronFlavour"]==0].mean()
df2[df2["Jet_hadronFlavour"]==4]=df2[df2["Jet_hadronFlavour"]==4].div(c_mu)
df2[df2["Jet_hadronFlavour"]==5]=df2[df2["Jet_hadronFlavour"]==5].div(b_mu)
df2[df2["Jet_hadronFlavour"]==0]=df2[df2["Jet_hadronFlavour"]==0].div(l_mu)

mean_prod_b=df2.groupby(level=0).prod().mean()
print(f"\n\nbTag mean on events (normalized): {mean_prod_b}")


#%%
c_sources=["Extrap", "Interp", "LHEScaleWeight_muF", "LHEScaleWeight_muR", "PSWeightFSR", "PSWeightISR", "PUWeight", "Stat", "XSec_BRUnc_DYJets_b", "XSec_BRUnc_DYJets_c", "XSec_BRUnc_WJets_c", "jer", "jesTotal"]

c_systs=[]

for c_source in c_sources:
    c_systs.append("up_"+c_source)
    c_systs.append("down_"+c_source)
    
    
arrays=evaluate_ctag(arrays,"central")
for c_syst in c_systs:
    arrays=evaluate_ctag(arrays,c_syst)



ctag_df=ak.to_dataframe(arrays)
c_systs=["c_"+c_syst for c_syst in c_systs]
cdf=ctag_df[c_systs+["c_central","Jet_hadronFlavour","Jet_btagDeepFlavB"]]





cdf=cdf[cdf["Jet_hadronFlavour"]!=-1]
c_cdf=cdf[cdf["Jet_hadronFlavour"]==4]
b_cdf=cdf[(cdf["Jet_hadronFlavour"]==5)]
l_cdf=cdf[(cdf["Jet_hadronFlavour"]==0)]

print(f'ctag \t means: (on hadFlav==4) \n----------------------------\n{c_cdf.mean()}')

print("\n##############################\n")

print(f'ctag means: (on hadFlav==5) \n----------------------------\n{b_cdf.mean()}')
print("\n##############################\n")
print(f'ctag means: (on hadFlav==0) \n----------------------------\n{l_cdf.mean()}')
print("\n")



#%%

print(f"cTag mean on events (not normalized):\n{cdf.groupby(level=0).prod().mean()}")

c_cmu=c_cdf.mean()
b_cmu=b_cdf.mean()
l_cmu=l_cdf.mean()
cdf[cdf["Jet_hadronFlavour"]==4]=cdf[cdf["Jet_hadronFlavour"]==4].div(c_cmu)
cdf[cdf["Jet_hadronFlavour"]==5]=cdf[cdf["Jet_hadronFlavour"]==5].div(b_cmu)
cdf[cdf["Jet_hadronFlavour"]==0]=cdf[cdf["Jet_hadronFlavour"]==0].div(l_cmu)

mean_prod_c=cdf.groupby(level=0).prod().mean()
print(f"\n\ncTag mean on events (normalized): {mean_prod_c}")



#%%


mplhep.style.use("CMS")

plt.figure()
f,(a0,a1)=plt.subplots(2,1,gridspec_kw={"hspace":0,'height_ratios': [3, 1]})

score=np.array(c_df["Jet_btagDeepFlavB"])
central=np.array(c_df["b_central"])
up=np.array(c_df["b_up_cferr1"])
down=np.array(c_df["b_down_cferr1"])
bins=40
c=a0.hist(score,weights=central,color="black",label="central",histtype="step",bins=bins)
u=a0.hist(score,weights=up/c_df.mean()["b_up_cferr1"],color="red",label="up_cferr1",histtype="step",bins=bins)
d=a0.hist(score,weights=down/c_df.mean()["b_down_cferr1"],color="dodgerblue",label="down_cferr1",histtype="step",bins=bins)
a0.legend()
a0.set_yscale("log")

a0.grid()

a0.set_ylabel("Counts")
a0.set_title("w=(cferr1/mean(cferr1))[hadFlav==4]")

bins=c[1][1:]+(c[1][1]-c[1][0])/2

a1.plot(bins,u[0]/c[0],"--r.",color="red",label="up_cferr1",markersize=8)
a1.plot(bins,d[0]/c[0],"--b.",color="dodgerblue",label="down_cferr1",markersize=8)
a1.plot([0,1],[1,1],color="black")
a1.grid()
a1.set_ylabel("Ratio")
plt.xlabel("DeepFlavB")



plt.figure()
f,(a0,a1)=plt.subplots(2,1,gridspec_kw={"hspace":0,'height_ratios': [3, 1]})

score=np.array(c_df["Jet_btagDeepFlavB"])
central=np.array(c_df["b_central"])
up=np.array(c_df["b_up_cferr1"])
down=np.array(c_df["b_down_cferr1"])
bins=40
c=a0.hist(score,weights=central,color="black",label="central",histtype="step",bins=bins)
u=a0.hist(score,weights=up,color="red",label="up_cferr1",histtype="step",bins=bins)
d=a0.hist(score,weights=down,color="dodgerblue",label="down_cferr1",histtype="step",bins=bins)
a0.legend()
a0.set_yscale("log")

a0.grid()

a0.set_ylabel("Counts")
a0.set_title("w=cferr1[hadFlav==4]")

bins=c[1][1:]+(c[1][1]-c[1][0])/2

a1.plot(bins,u[0]/c[0],"--r.",color="red",label="up_cferr1",markersize=8)
a1.plot(bins,d[0]/c[0],"--b.",color="dodgerblue",label="down_cferr1",markersize=8)
a1.plot([0,1],[1,1],color="black")
a1.grid()
a1.set_ylabel("Ratio")
plt.xlabel("DeepFlavB")




#%%

