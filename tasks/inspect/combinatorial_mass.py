#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep
import copy
import json

def np_and(*args):
    res=args[0]
    for arg in args[1:]:
        res=np.bitwise_and(res,arg)
    return res

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
    return cut3


events=NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal/signal_train.root",
    schemaclass=NanoAODSchema
).events()

events=Muon_cuts(events)

#%%
#! HADW
lhe_W1=events.LHEPart[:,[3,4]]
lhe_W2=events.LHEPart[:,[6,7]]
lhe_W1=lhe_W1[ak.all(np.abs(lhe_W1.pdgId)<6,axis=1)]
lhe_W2=lhe_W2[ak.all(np.abs(lhe_W2.pdgId)<6,axis=1)]
lhe_W=ak.concatenate([lhe_W1,lhe_W2],axis=0)



jet_W,dR=lhe_W.nearest(events.Jet,axis=1,return_metric=True)
jet_W=jet_W[dR<0.4]
mask=(ak.num(jet_W)==2)
jet_W=jet_W[ak.num(jet_W)==2]

jet_other=events.Jet[mask][np_and(jet_W[:,0].pt!=events.Jet[mask].pt,
                                jet_W[:,1].pt!=events.Jet[mask].pt)]

jet_other=ak.concatenate([jet_other,events.Jet[~mask]],axis=0)

#%%
Wmass=np.asarray((jet_W[:,0]+jet_W[:,1]).mass)

other_mass=ak.combinations(jet_other,2)
other_mass=np.asarray(ak.flatten((other_mass["0"]+other_mass["1"]).mass))

#%%
mplhep.style.use("CMS")

fig, ax = plt.subplots()


ax.hist(Wmass,bins=100,range=(0,500),label="HadW mass",histtype="step",density=True,linewidth=3)
ax.hist(other_mass,bins=100,range=(0,500),label="Combinatorial bkg",histtype="step",density=True,linewidth=3)
ax.set_xlabel("HadW Mass [GeV]")
ax.set_ylabel("Density")
#ax.set_yscale("log")
ax.legend()
ax.grid()
mplhep.cms.text("Private Work",ax=ax)