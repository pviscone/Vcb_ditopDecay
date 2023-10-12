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
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2820000/C6DC1D11-72BB-8C45-8625-0B874FB177C2.root",
    schemaclass=NanoAODSchema
).events()


TTJets_diHad = NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2520000/3D7C0328-DC6C-C046-8DF4-C7024E774FBA.root",
    schemaclass=NanoAODSchema
).events()

TTJets_diHad = TTJets_diHad[
    np_and(
        np.abs(TTJets_diHad.GenPart[TTJets_diHad.GenPart.pdgId==-24][:,-1].children.pdgId[:,0])<6,
        np.abs(TTJets_diHad.GenPart[TTJets_diHad.GenPart.pdgId==24][:,-1].children.pdgId[:,0])<6
    )
]


WJets_toLNu=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv2/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/230000/87A6BF8E-A6CC-0549-ADC0-260011EE2894.root",
    schemaclass=NanoAODSchema
).events()

#!There are negative genweights in this sample
WWJets_LNuQQ=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/60000/43340426-80F1-534C-9D61-5F0D90AD57B3.root",
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
    "/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/others/C702A2B0-292B-854F-96F5-CAF931254A40.root",
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
    #"WWJets_LNuQQ":WWJets_LNuQQ
}



def Muon_cuts(events):
    n_tot=len(events)
    events["Muon"]=events.Muon[events.Muon.looseId & (events.Muon.pfIsoId>=1)]
    events=events[ak.num(events.Muon)>=1]
    events["Muon"]=events.Muon[:,0]
    cut1=events[np_and(events.Muon.pt>=26,
                        events.Muon.eta<2.4,)]
    
    cut1_n=len(cut1)
    cut2=copy.copy(cut1)
    cut2["Jet"]=cut2.Jet[np_and(
                                    cut2.Jet.jetId>0,
                                    cut2.Jet.puId>0,
                                    cut2.Jet.pt>20,
                                    np.abs(cut2.Jet.eta)<4.8,
                                    cut2.Jet.delta_r(cut2.Muon)>0.4)]

    cut2=cut2[ak.num(cut2.Jet)>=4]
    cut2_n=len(cut2)
    cut3=copy.copy(cut2)
    cut3=cut3[(ak.max(cut3.Jet.btagDeepFlavCvL,axis=1)>0.2793)]
    cut3_n=len(cut3)
    return {
            "1 Loose $\mu $ \n $ p_{T}>26, |\eta|<2.4$":cut1_n/n_tot,
            "4 Jets \n $ p_{T}>20, |\eta|<4.8$":cut2_n/n_tot,
            "1 Medium bTag":cut3_n/n_tot}
        
    

def Electron_cuts(events):
    n_tot=len(events)
    events["Electron"]=events.Electron[events.Electron.mvaFall17V2noIso_WP90]
    events=events[ak.num(events.Electron)>=1]
    events["Electron"]=events.Electron[:,0]
    cut1=events[np_and(events.Electron.pt>=30,
                        events.Electron.eta<2.4,)]
    
    cut1_n=len(cut1)
    cut2=copy.copy(cut1)
    cut2["Jet"]=cut2.Jet[np_and(
                                    cut2.Jet.jetId>0,
                                    cut2.Jet.puId>0,
                                    cut2.Jet.pt>20,
                                    np.abs(cut2.Jet.eta)<4.8,
                                    cut2.Jet.delta_r(cut2.Electron)>0.4)]

    cut2=cut2[ak.num(cut2.Jet)>=4]
    cut2_n=len(cut2)
    cut3=copy.copy(cut2)
    cut3=cut3[(ak.max(cut3.Jet.btagDeepFlavCvL,axis=1)>0.2793)]
    cut3_n=len(cut3)
    return {
            "1 Medium $e$ \n $ p_{T}>30, |\eta|<2.4$":cut1_n/n_tot,
            "4 Jets \n$ p_{T}>20, |\eta|<4.8$":cut2_n/n_tot,
            "1 Medium bTag":cut3_n/n_tot}
    
    
    
    
eff_dict={
    "Muons":{
        "1 Loose $\mu $ \n $ p_{T}>26, |\eta|<2.4$":{},
        "4 Jets \n $ p_{T}>20, |\eta|<4.8$":{},
        "1 Medium bTag":{}
        },
    "Electrons":{
        "1 Medium $e$ \n $ p_{T}>30, |\eta|<2.4$":{},
        "4 Jets \n$ p_{T}>20, |\eta|<4.8$":{},
        "1 Medium bTag":{}
        }
    }



for dataset in datasets:
    eff_mu=Muon_cuts(datasets[dataset])
    eff_ele=Electron_cuts(datasets[dataset])
    if "semiLept" in dataset:
        lumi=sample_dict["bkg"]["nEv_lumi"]*0.33
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
    #"WWJets_LNuQQ":WWJets_LNuQQ
}




mplhep.style.use("CMS")
for region in ["Muons","Electrons"]:
    fig,ax=plt.subplots(1,1)
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
    mplhep.cms.lumitext("137 fb$^{-1}$ (13 TeV)",ax=ax)
