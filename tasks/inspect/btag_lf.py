#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import mplhep
import matplotlib as mpl
import correctionlib

print("REMEMBER TO USE THE COFFEA ENVIROMENT")

ceval = correctionlib.CorrectionSet.from_file("../combine/systematics/json/btagging.json")

btag = (ceval["deepJet_shape"])


def np_and(*args):
    out=args[0]
    for i in range(1,len(args)):
        out=np.bitwise_and(out,args[i])
    return out

def np_or(*args):
    out=args[0]
    for i in range(1,len(args)):
        out=np.bitwise_or(out,args[i])
    return out

def eval(events,name):

    jet_pt=np.asarray(ak.flatten(events.Jet.pt))
    jet_eta=np.abs(np.asarray(ak.flatten(events.Jet.eta)))
    jet_hf=np.asarray(ak.flatten(events.Jet.hadronFlavour))
    jet_btag=np.asarray(ak.flatten(events.Jet.btagDeepFlavB))
    
    np_weight=np.ones_like(jet_pt)
    
    
    if "cferr" in name:
        mask=jet_hf==4
    else:
        mask=np_or(jet_hf==5,jet_hf==0)
    
    mask=np_and(mask, jet_eta<2.5)
    
    
    np_weight[mask]=(btag.evaluate(name,
                                   jet_hf[mask],
                                   jet_eta[mask],
                                   jet_pt[mask],
                                   jet_btag[mask])/
                     btag.evaluate("central",
                                   jet_hf[mask],
                                   jet_eta[mask],
                                   jet_pt[mask],
                                   jet_btag[mask])
                     )
    
    events["Jet",name]=ak.ones_like(events.Jet.pt)
    dummy_musk=np.ones_like(jet_pt)>0
    np.asarray(ak.flatten(events["Jet",name]))[dummy_musk]=np_weight
    prod=ak.prod(events["Jet",name],axis=1)
    events[f"Ev_{name}"]=prod/np.mean(np.asarray((prod)))
    return events


def create(file_name,n):

    events = NanoEventsFactory.from_root(
        file_name,
        schemaclass=NanoAODSchema.v6,
    ).events()
    
    events=events[ak.any(np.abs(events.LHEPart.pdgId)==13,axis=1)]
    events=events[:n]

    events["Muon"]=events.Muon[events.Muon.looseId & (events.Muon.pfIsoId>=1)]

    events=events[ak.num(events.Muon)>=1]
    events["Muon"]=events.Muon[:,0]

    events["Jet"]=events.Jet[np_and(
                                    events.Jet.jetId>0,
                                    events.Jet.puId>0,
                                    events.Jet.pt>20,
                                    np.abs(events.Jet.eta)<4.8,
                                    events.Jet.delta_r(events.Muon)>0.4)]

    events=events[ak.num(events.Jet)>=4]

    events=events[np_and(events.Muon.pt>=26,
                        events.Muon.eta<2.4,)]

    events=eval(events,"up_lf")
    events=eval(events,"down_lf")

    events=events[ak.max(events.Jet.btagDeepFlavB,axis=1)>=0.2793]
    return events
#%%


bkg_name="2A83E205-B7CE-3744-86A0-5E8B1E807D44.root"
signal_name="/scratchnvme/pviscone/Preselection_Skim/signal/BigMuons.root"

n=100000
bkg=create(bkg_name,n)
signal=create(signal_name,n)


n_tot=np.min([len(bkg),len(signal)])
bkg=bkg[:n_tot]
signal=signal[:n_tot]

#%%

weird_sig=signal[np_or(signal.Ev_down_lf>1.2,
                    signal.Ev_down_lf<0.8,)]

weird_bkg=bkg[np_or(bkg.Ev_down_lf>1.2,
                    bkg.Ev_down_lf<0.8,)]



print(f"ev_signal: {len(signal)}")
print(f"ev_bkg: {len(bkg)}")
print(f"weird_sig: {len(weird_sig)}")
print(f"weird_bkg: {len(weird_bkg)}")


weird_sig_jets=weird_sig.Jet[np_or(
                                weird_sig.Jet.down_lf>1.2,
                                weird_sig.Jet.down_lf<0.8,)]

weird_bkg_jets=weird_bkg.Jet[np_or(
                                weird_bkg.Jet.down_lf>1.2,
                                weird_bkg.Jet.down_lf<0.8,)]





#%%
plt.figure()
plt.title("hadron flavour")
plt.hist(np.asarray(ak.flatten(weird_sig_jets.hadronFlavour)),range=(0,6),bins=6,histtype="step",label="signal")
plt.hist(np.asarray(ak.flatten(weird_bkg_jets.hadronFlavour)),range=(0,6),bins=6,histtype="step",label="bkg")
plt.legend()

plt.figure()
plt.title("btag")
plt.hist(np.asarray(ak.flatten(weird_sig_jets.btagDeepFlavB)),range=(0,1),bins=20,histtype="step",label="signal")
plt.hist(np.asarray(ak.flatten(weird_bkg_jets.btagDeepFlavB)),range=(0,1),bins=20,histtype="step",label="bkg")
plt.legend()

plt.figure()
plt.title("eta")
plt.hist(np.abs(np.asarray(ak.flatten(np.abs(weird_sig_jets.eta)))),range=(0,2.5),bins=20,histtype="step",label="signal")
plt.hist(np.abs(np.asarray(ak.flatten(np.abs(weird_bkg_jets.eta)))),range=(0,2.5),bins=20,histtype="step",label="bkg")
plt.legend()
plt.figure()
plt.title("pt")
plt.hist(np.asarray(ak.flatten(weird_sig_jets.pt)),range=(0,200),bins=20,histtype="step",label="signal")
plt.hist(np.asarray(ak.flatten(weird_bkg_jets.pt)),range=(0,200),bins=20,histtype="step",label="bkg")
plt.legend()