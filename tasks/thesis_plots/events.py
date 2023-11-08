


#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
import numpy as np
import awkward as ak

sys.path.append("../../utils/coffea_utils")
from coffea_utils import Electron_cuts, Muon_cuts, Jet_parton_matching,np_and,np_or

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


WWJets_LNuQQ=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/WWJets/WWJets.root",
    schemaclass=NanoAODSchema
).events()



signal=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/dummy_signal/signal.root",
    schemaclass=NanoAODSchema
).events()

semiLept=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/powheg/root_files/others/powheg.root",
    schemaclass=NanoAODSchema
).events()



tW=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/tW/tw.root",
    schemaclass=NanoAODSchema
).events()

#SL_mask=ak.num(tW.LHEPart.pdgId[np_and(np.abs(tW.LHEPart.pdgId)>=11,np.abs(tW.LHEPart.pdgId)<21)])<=2

#tW=tW[SL_mask]

tQ=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/others/tq/tq.root",
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
    "WJets": 8.5e9,
    "WWJets":7.1e6,
    "tW":4.8e6,
    "tq":9.6e6
}


weights={}


def ev_weight(event,n_ev):
    return (event.genWeight)*n_ev/np.sum(event.genWeight)


for key in datasets.keys():
    weights[key]=ev_weight(datasets[key],n_ev[key])
    
    
    
def obj_selection(events):
    #Muons
    events["Muon"]=events.Muon[np_and(events.Muon.pt>26,
                                      events.Muon.looseId,
                                      events.Muon.pfIsoId > 1,
                                      np.abs(events.Muon.eta)<2.4)]
    
    #Electrons
    events["Electron"]=events.Electron[np_and(events.Electron.pt>30,
                                              events.Electron.mvaFall17V2Iso_WP90,
                                              np.abs(events.Electron.eta)<2.4)]
    
    
    #Jets
    events["Jet"]=events.Jet[np_and(events.Jet.pt>20,
                                    np.abs(events.Jet.eta)<4.8,
                                    events.Jet.jetId>0,
                                    events.Jet.puId > 0)]
    
    lept=ak.concatenate([events.Electron,events.Muon],axis=1)
    lept_argsort=ak.argsort(lept.pt,axis=1,ascending=False)
    lept=lept[lept_argsort]
    lept=ak.pad_none(lept,1,clip=True)
    jet_lept_dR=events.Jet.delta_r(lept)
    jet_lept_dR=ak.fill_none(jet_lept_dR,999)
    
    events["Jet"]=events.Jet[jet_lept_dR>0.4]
    return events



def event_selection(events, region,weights):
    assert region in ["Muon","Electron","both"]
    
    if region=="Muon":
        mask=ak.count(events.Muon.pt,axis=1)>=1
        events=events[mask]
        events["Muon"]=events.Muon[:,0]
        weights=weights[mask]
    elif region=="Electron":
        mask=ak.count(events.Electron.pt,axis=1)>=1
        events=events[mask]
        events["Electron"]=events.Electron[:,0]
        weights=weights[mask]
    elif region=="both":
        mask=np_or(ak.count(events.Muon.pt,axis=1)>=1,
                   ak.count(events.Electron.pt,axis=1)>=1)
        events=events[mask]
        weights=weights[mask]

    mask=ak.count(events.Jet.pt,axis=1)>=4
    events=events[ak.count(events.Jet.pt,axis=1)>=4]
    weights=weights[mask]
    return events[~ak.is_none(events)],weights[~ak.is_none(events)]


def btag_selection(events,weights):
    mask=ak.max(events.Jet.btagDeepFlavB,axis=1)>0.277
    events=events[mask]
    weights=weights[mask]
    return events[~ak.is_none(events)],weights[~ak.is_none(events)]


# %%
