


#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import mplhep
import dataframe_image as dfi

sys.path.append("../../utils/coffea_utils")
from coffea_utils import Electron_cuts, Muon_cuts, Jet_parton_matching,np_and,np_or

#%%

#! REMEMBER TO VOMS-PROXY-INIT BEFORE RUNNING THIS SCRIPT
TTJets_diLept = NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2820000/C6DC1D11-72BB-8C45-8625-0B874FB177C2.root",
    schemaclass=NanoAODSchema
).events()


TTJets_diHad = NanoEventsFactory.from_root(
    "root://xrootd-cms.infn.it//store/mc/RunIIAutumn18NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/110000/3EC0AD49-F86C-9649-BDC4-E9217904FCC0.root",
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

signal=ak.concatenate([TT_semilept_cb_Electron,TT_semilept_cb_Muon,TT_semilept_cb_Tau])


TT_Jets_LNuQQ_NoCKM=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/others/C702A2B0-292B-854F-96F5-CAF931254A40.root",
    schemaclass=NanoAODSchema
).events()

TT_Jets_LNuQQ_NoCKM_Electron=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==11),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==11))
    ]
TT_Jets_LNuQQ_NoCKM_Muon=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==13),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==13))
    ]
TT_Jets_LNuQQ_NoCKM_Tau=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==15),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==15))
    ]

semiLept=TT_Jets_LNuQQ_NoCKM


tW=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer19UL18NanoAODv2/ST_tW_antitop_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/280000/69896C66-7F8E-904A-B3DF-A3E82970B6EE.root",
    schemaclass=NanoAODSchema
).events()

SL_mask=ak.num(tW.LHEPart.pdgId[np_and(np.abs(tW.LHEPart.pdgId)>=11,
            np.abs(tW.LHEPart.pdgId)<21)])<=2

tW=tW[SL_mask]

tQ=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer16NanoAODv7/ST_t-channel_top_5f_leptDecays_TuneCUETP8M1_13TeV-powhegV2-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/120000/788061AB-FE17-D340-A686-E8F125F5A28F.root",
    schemaclass=NanoAODSchema
).events()





datasets = {
    "signal": signal,
    "TTsemiLept": semiLept,
    "TTdiLept": TTJets_diLept,
    "TTdiHad": TTJets_diHad,
    "WJets": WJets_toLNu,
    "WWJets":WWJets_LNuQQ,
    "tW":tW,
    "tq":tQ
}

n_ev={
    "signal": 4.2e4,
    "TTsemiLept": 5.0e7,
    "TTdiLept": 1.2e7,
    "TTdiHad": 5.2e7,
    "WJets": 2.6e9,
    "WWJets":1.7e6,
    "tW":4.8e6,
    "tq":9.6e6
}


weights={}


def ev_weight(event,n_ev):
    return np.sign(event.genWeight)*n_ev/len(event.genWeight)


for key in datasets.keys():
    weights[key]=ev_weight(datasets[key],n_ev[key])
    
    
    
def obj_selection(events):
    #Muons
    events["Muon"]=ak.mask(events.Muon,np_and(events.Muon.pt>26,
                                      events.Muon.looseId,
                                      events.Muon.pfIsoId > 1,
                                      np.abs(events.Muon.eta)<2.4))
    
    #Electrons
    events["Electron"]=ak.mask(events.Electron,np_and(events.Electron.pt>30,
                                              events.Electron.mvaFall17V2Iso_WP90,
                                              np.abs(events.Electron.eta)<2.4))
    
    
    #Jets
    events["Jet"]=ak.mask(events.Jet,np_and(events.Jet.pt>20,
                                    np.abs(events.Jet.eta)<4.8,
                                    events.Jet.jetId>0,
                                    events.Jet.puId > 0))
    
    lept=ak.concatenate([events.Electron,events.Muon],axis=1)
    lept_argsort=ak.argsort(lept.pt,axis=1,ascending=False)
    lept=lept[lept_argsort]
    lept=ak.pad_none(lept,1)
    jet_lept_dR=events.Jet.delta_r(lept[:,0])
    
    events["Jet"]=ak.mask(events.Jet,jet_lept_dR>0.4)
    return events



def event_selection(events, region):
    assert region in ["Muon","Electron"]
    
    n_ev_before=len(events)
    if region=="Muon":
        events=ak.mask(events,ak.num(events.Muon)>=1)
        events["Muon"]=events.Muon[:,0]
    else:
        events=ak.mask(events,ak.num(events.Electron)>=1)
        events["Electron"]=events.Electron[:,0]
    
    events=ak.mask(events,np_and(ak.num(events.Jet)>=4,
                         ak.max(events.Jet.btagDeepFlavB,axis=1)>=0.2793))
    n_ev_after=len(events)
    return events,n_ev_after/n_ev_before



