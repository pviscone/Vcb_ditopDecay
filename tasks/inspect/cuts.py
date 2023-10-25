#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep
import copy
import json

sys.path.append("../../utils/coffea_utils")
from coffea_utils import np_and,np_or

sample_dict=json.load(open("../combine/systematics/json/samples.json","r"))

#%%
#! REMEMBER TO VOMS-PROXY-INIT BEFORE RUNNING THIS SCRIPT
TTJets_diLept = NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/diLept/root_files/predict/diLept.root",
    schemaclass=NanoAODSchema
).events()


TTJets_diHad = NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/diHad/root_files/diHad.root",
    schemaclass=NanoAODSchema
).events()


WJets_toLNu=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/WJets/root_files/WJets.root",
    schemaclass=NanoAODSchema
).events()


WWJets=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/WWJets/43340426-80F1-534C-9D61-5F0D90AD57B3.root",
    schemaclass=NanoAODSchema
).events()



TT_semilept_cb_Electron=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal_Electrons/signal_Electrons_test.root",
    schemaclass=NanoAODSchema
).events()

TT_semilept_cb_Muon=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal/signal_test.root",
    schemaclass=NanoAODSchema
).events()

TT_semilept_cb_Tau=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal_Tau/signal_tau.root",
    schemaclass=NanoAODSchema
).events()



TT_Jets_LNuQQ_NoCKM=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/others/powheg.root",
    schemaclass=NanoAODSchema
).events()

TT_Jets_LNuQQ_NoCKM_Electron=TT_Jets_LNuQQ_NoCKM[
    ak.any(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId)==11,axis=1)
    ]
TT_Jets_LNuQQ_NoCKM_Muon=TT_Jets_LNuQQ_NoCKM[
    ak.any(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId)==13,axis=1)
    ]
TT_Jets_LNuQQ_NoCKM_Tau=TT_Jets_LNuQQ_NoCKM[
    ak.any(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId)==15,axis=1)
    ]


tW=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/tW/69896C66-7F8E-904A-B3DF-A3E82970B6EE.root",
    schemaclass=NanoAODSchema
).events()

SL_mask=ak.num(tW.LHEPart.pdgId[np_and(np.abs(tW.LHEPart.pdgId)>=11,
            np.abs(tW.LHEPart.pdgId)<21)])<=2

tW=tW[SL_mask]

tQ=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/tq/788061AB-FE17-D340-A686-E8F125F5A28F.root",
    schemaclass=NanoAODSchema
).events()




datasets = {
    "signalEle": TT_semilept_cb_Electron,
    "signalMu": TT_semilept_cb_Muon,
    "signalTau": TT_semilept_cb_Tau,
    "semiLeptEle": TT_Jets_LNuQQ_NoCKM_Electron,
    "semiLeptMu": TT_Jets_LNuQQ_NoCKM_Muon,
    "semiLeptTau": TT_Jets_LNuQQ_NoCKM_Tau,
    "diLept": TTJets_diLept,
    "diHad": TTJets_diHad,
    "WJets": WJets_toLNu,
    "WWJets":WWJets,
    "tW":tW,
    "tq":tQ
}



def Muon_cuts(events):
    n_tot=ak.sum(events.genWeight)
    events["Muon"]=events.Muon[events.Muon.looseId & (events.Muon.pfIsoId>=1)]
    events=events[ak.num(events.Muon)>=1]
    events["Muon"]=events.Muon[:,0]
    cut1=events[np_and(events.Muon.pt>=26,
                        events.Muon.eta<2.4,)]
    
    cut1_n=ak.sum(cut1.genWeight)
    cut2=copy.copy(cut1)
    cut2["Jet"]=cut2.Jet[np_and(
                                    cut2.Jet.jetId>0,
                                    cut2.Jet.puId>0,
                                    cut2.Jet.pt>20,
                                    np.abs(cut2.Jet.eta)<4.8,
                                    cut2.Jet.delta_r(cut2.Muon)>0.4)]

    cut2=cut2[ak.num(cut2.Jet)>=4]
    cut2_n=ak.sum(cut2.genWeight)
    cut3=copy.copy(cut2)
    cut3=cut3[(ak.max(cut3.Jet.btagDeepFlavB,axis=1)>0.2793)]
    cut3_n=ak.sum(cut3.genWeight)
    return {
            "$\geq 1\mu $ \nLoose\n $ p_{T}>26$\n$|\eta|<2.4$":cut1_n/n_tot,
            "$\geq 4$ Jets\n$ p_{T}>20$\n$|\eta|<4.8$":cut2_n/n_tot,
            "1 Medium\nbTag":cut3_n/n_tot}
        
    

def Electron_cuts(events):
    n_tot=ak.sum(events.genWeight)
    events["Electron"]=events.Electron[events.Electron.mvaFall17V2noIso_WP90]
    events=events[ak.num(events.Electron)>=1]
    events["Electron"]=events.Electron[:,0]
    cut1=events[np_and(events.Electron.pt>=30,
                        events.Electron.eta<2.4,)]
    
    cut1_n=ak.sum(cut1.genWeight)
    cut2=copy.copy(cut1)
    cut2["Jet"]=cut2.Jet[np_and(
                                    cut2.Jet.jetId>0,
                                    cut2.Jet.puId>0,
                                    cut2.Jet.pt>20,
                                    np.abs(cut2.Jet.eta)<4.8,
                                    cut2.Jet.delta_r(cut2.Electron)>0.4)]

    cut2=cut2[ak.num(cut2.Jet)>=4]
    cut2_n=ak.sum(cut2.genWeight)
    cut3=copy.copy(cut2)
    cut3=cut3[(ak.max(cut3.Jet.btagDeepFlavB,axis=1)>0.2793)]
    cut3_n=ak.sum(cut3.genWeight)
    return {
            "$\geq 1e$\nMedium\n$ p_{T}>30$\n$|\eta|<2.4$":cut1_n/n_tot,
            "$\geq 4$ Jets\n$ p_{T}>20$\n$|\eta|<4.8$":cut2_n/n_tot,
            "1 Medium\nbTag":cut3_n/n_tot}
    
    
    
    
eff_dict={
    "Muons":{
        "$\geq 1\mu $ \nLoose\n $ p_{T}>26$\n$|\eta|<2.4$":{},
        "$\geq 4$ Jets\n$ p_{T}>20$\n$|\eta|<4.8$":{},
        "1 Medium\nbTag":{}
        },
    "Electrons":{
        "$\geq 1e$\nMedium\n$ p_{T}>30$\n$|\eta|<2.4$":{},
        "$\geq 4$ Jets\n$ p_{T}>20$\n$|\eta|<4.8$":{},
        "1 Medium\nbTag":{}
        }
    }



for dataset in datasets:
    eff_mu=Muon_cuts(datasets[dataset])
    eff_ele=Electron_cuts(datasets[dataset])
    if "semiLept" in dataset:
        lumi=sample_dict["bkg"]["nEv_lumi"]*0.33
    elif dataset=="tW":
        lumi=4.8e6
    elif dataset=="tq":
        lumi=9.6e6
    elif dataset=="WWJets":
        lumi=7.1e6
    elif dataset=="WJets":
        lumi=8.5e9
    else:
        lumi=sample_dict[dataset]["nEv_lumi"]
        
    for key in eff_mu:
        eff_dict["Muons"][key][dataset]=eff_mu[key]*lumi
    for key in eff_ele:
        eff_dict["Electrons"][key][dataset]=eff_ele[key]*lumi
        
#%%


colors = {
    "signalEle": "dodgerblue",
    "signalMu": "red",
    "signalTau": "green",
    "semiLeptEle": "mediumturquoise",
    "semiLeptMu": "salmon",
    "semiLeptTau": "lime",
    "diLept":"darkblue",
    "diHad": "pink",
    "WJets": "orange",
    "WWJets":"#4376c9",
    "tW":"plum",
    "tq":"#91d36e"
}




mplhep.style.use("CMS")
for region in ["Muons","Electrons"]:
    fig,ax=plt.subplots(1,1,figsize=(16,9))
    ax.set_xlabel("Events")
    ax.grid()
    lab=None
    for i,cut in enumerate(reversed(eff_dict[region])):
        bar_width=0
        sorted_keys = sorted(eff_dict[region][cut], key=eff_dict[region][cut].get)
        for dataset in sorted_keys:
            if i==0:
                num_exp = "{:.1e}".format(float(eff_dict[region][cut][dataset])).replace('e', '$\cdot 10^').replace('+0', '')
                lab=f"{dataset} ({num_exp}$)"
            else:
                lab=None
            ax.barh(cut,eff_dict[region][cut][dataset],left=bar_width,label=lab,color=colors[dataset])
            bar_width+=eff_dict[region][cut][dataset]
    ax.legend(loc="lower right",ncol=1)
    #ax.set_title(region)
    ax.set_xscale("log")
    ax.set_xlim(1e2,1e14)
    ax.set_axisbelow(True)
    mplhep.cms.text("Private Work",loc=0,ax=ax)
    mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax)

# %%
