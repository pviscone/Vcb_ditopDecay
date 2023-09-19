#%%


import uproot
import correctionlib
import numpy as np
import pandas as pd
import awkward as ak
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
import copy
import matplotlib.pyplot as plt
import mplhep
from pprint import pprint

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

def evaluate_ctag(array,name):
    array[name]=ak.ones_like(array["Jet_pt"])
    array[name]=(ctag(name,
                    array["Jet_hadronFlavour"],
                    array["Jet_btagDeepFlavCvL"],
                    array["Jet_btagDeepFlavCvB"])/
                ctag("central",
                    array["Jet_hadronFlavour"],
                    array["Jet_btagDeepFlavCvL"],
                    array["Jet_btagDeepFlavCvB"]))
    return array


f=uproot.open("/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/predict/3A361FB1-E533-1C4D-9EA7-ACFA17F05B69.root")


n_ev=1000000
arrays=f["Events"].arrays(["Jet_pt","Jet_eta","Jet_btagDeepFlavB","Jet_btagDeepFlavCvL","Jet_btagDeepFlavCvB","Jet_hadronFlavour","Jet_jetId","Jet_puId"])[:n_ev,:7]
mu_arrays=f["Events"].arrays(["Muon_looseId","Muon_pfIsoId","Muon_pt","Muon_eta"])[:n_ev]


arrays=arrays[np_and(
                    arrays["Jet_pt"]>20,
                    np.abs(arrays["Jet_eta"])<2.5,
                    arrays["Jet_jetId"]>0,
                    arrays["Jet_puId"]>0,
                    )
              ]

mu_arrays=mu_arrays[np_and(mu_arrays["Muon_looseId"],
                    mu_arrays["Muon_pfIsoId"]>1,
                    np.abs(mu_arrays["Muon_eta"])<2.4,)
]


cuts=ak.fill_none(np_and(ak.num(mu_arrays["Muon_pt"])>=1,
       ak.max(mu_arrays["Muon_pt"],axis=1)>26,
       ak.num(arrays["Jet_pt"])>=4,
       ak.max(arrays["Jet_btagDeepFlavB"],axis=1)>0.2770,
       ),False)

arrays=arrays[cuts]

#%%
#!ctag corrections
c_sources=["Extrap", "Interp", "LHEScaleWeight_muF", "LHEScaleWeight_muR", "PSWeightFSR", "PSWeightISR", "PUWeight", "Stat", "XSec_BRUnc_DYJets_b", "XSec_BRUnc_DYJets_c", "XSec_BRUnc_WJets_c", "jer", "jesTotal"]

c_systs=["central"]
for c_source in c_sources:
    c_systs.append("up_"+c_source)
    c_systs.append("down_"+c_source)
    
for c_syst in c_systs:
    arrays=evaluate_ctag(arrays,c_syst)
#%%
#! Compute means on hadron flavours
corr_dict={}
for had_flav in [0,4,5]:
    corr_dict[str(had_flav)]={}
    for c_syst in c_systs:
        m_value=np.mean(arrays[arrays["Jet_hadronFlavour"]==had_flav][c_syst])
        corr_dict[str(had_flav)][c_syst]=m_value
        
        had_mask=np.asarray(ak.flatten(arrays["Jet_hadronFlavour"]))==had_flav
        np.asarray(ak.flatten(arrays[c_syst]))[had_mask]=np.asarray(ak.flatten(arrays[c_syst]))[had_mask]/m_value
        

#%%
for c_syst in c_systs:
    m=np.mean(ak.prod(arrays[c_syst],axis=1))
    print(f"syst:{c_syst} \t {m}")
    
    
#%%
#!!!!!!!!!!!!!!!!!!!!!
#!BTAG CORRECTIONS
#!!!! DONT WORK WHY?


def evaluate_btag(array,name):
    array[f"b_{name}"]=ak.ones_like(array["Jet_pt"])
    mask=np.asarray(ak.flatten(array["Jet_hadronFlavour"]))>-1
    if name=="central":
        pass
    elif name.split("_")[1]=="cferr1" or name.split("_")[1]=="cferr2":
        mask=np_and(mask,np.asarray(ak.flatten(array["Jet_hadronFlavour"]))==4)
    else:
        mask=np_and(mask,np.asarray(ak.flatten(array["Jet_hadronFlavour"]))!=4)

    #mask=np_and(mask,np.asarray(ak.flatten(np.abs(array["Jet_eta"])))<2.5)
    
    np.asarray(ak.flatten(array["b_"+name]))[mask]=(btag(name,
                                                    np.asarray(ak.flatten(array["Jet_hadronFlavour"]))[mask],
                                                    np.asarray(ak.flatten(np.abs(array["Jet_eta"])))[mask],
                                                    np.asarray(ak.flatten(array["Jet_pt"]))[mask],
                                                    np.asarray(ak.flatten(array["Jet_btagDeepFlavB"]))[mask])/
                                                    btag("central",
                                                    np.asarray(ak.flatten(array["Jet_hadronFlavour"]))[mask],
                                                    np.asarray(ak.flatten(np.abs(array["Jet_eta"])))[mask],
                                                    np.asarray(ak.flatten(array["Jet_pt"]))[mask],
                                                    np.asarray(ak.flatten(array["Jet_btagDeepFlavB"]))[mask]))
    
    
    
    return array

#%%
b_sources=["hf","lf","hfstats1","hfstats2","lfstats1","lfstats2","cferr1","cferr2"]
b_systs=["central"]
for b_source in b_sources:
    b_systs.append("up_"+b_source)
    b_systs.append("down_"+b_source)
    
for b_syst in b_systs:
    arrays=evaluate_btag(arrays,b_syst)

#! Compute means on hadron flavours
corr_dict={}
for had_flav in [0,4,5]:
    corr_dict[str(had_flav)]={}
    for b_syst in b_systs:
        m_value=np.mean(arrays[arrays["Jet_hadronFlavour"]==had_flav][f"b_{b_syst}"])
        print(m_value)
        corr_dict[str(had_flav)][f"b_{b_syst}"]=m_value
        
        had_mask=np.asarray(ak.flatten(arrays["Jet_hadronFlavour"]))==had_flav
        np.asarray(ak.flatten(arrays[f"b_{b_syst}"]))[had_mask]=np.asarray(ak.flatten(arrays[f"b_{b_syst}"]))[had_mask]/m_value
        
for b_syst in b_systs:
    m=np.mean(ak.prod(arrays[f"b_{b_syst}"],axis=1))
    print(f"syst:{b_syst} \t {m}")
    