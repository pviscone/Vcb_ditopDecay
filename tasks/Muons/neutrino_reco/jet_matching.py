#%% Imports
import sys

sys.path.append("../../../utils")
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from histogrammer import Histogrammer
import mplhep as hep
import vector
import awkward as ak


plt.style.use(hep.style.CMS)
signal = uproot.open(
    "../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root")["Events"]


def get(key, numpy=False, library="ak"):
    arr = signal.arrays(key, library=library)[key]
    if numpy == True:
        return arr.to_numpy()
    else:
        return arr


def deltaPhi(phi1, phi2):
    dphi = (phi1 - phi2)
    dphi[dphi > np.pi] = 2*np.pi - dphi[dphi > np.pi]
    dphi[dphi < -np.pi] = -2*np.pi - dphi[dphi < -np.pi]
    return dphi

def deltaR(eta1,eta2,phi1,phi2):
    return np.sqrt((eta1-eta2)**2+deltaPhi(phi1,phi2)**2)

#%%
pdgId_Wdecay= get("LHEPart_pdgId")[:,[3,6]].to_numpy()

LHEmask = np.bitwise_or(pdgId_Wdecay ==13, pdgId_Wdecay == -13)

LeptB_LHE_phi= get("LHEPart_phi")[:,[2,5]].to_numpy()[LHEmask]
LeptB_LHE_eta= get("LHEPart_eta")[:,[2,5]].to_numpy()[LHEmask]
LeptB_LHE_pt= get("LHEPart_pt")[:,[2,5]].to_numpy()[LHEmask]

LeptB_LHE_4Vect=vector.awk({"pt":LeptB_LHE_pt,"eta":LeptB_LHE_eta,"phi":LeptB_LHE_phi,"mass":4.2*np.ones_like(LeptB_LHE_pt)})

Jet_pt = get("Jet_pt")
Jet_eta = get("Jet_eta")
Jet_phi = get("Jet_phi")
Jet_mass= get("Jet_mass")

Jet_4Vect=vector.awk({"pt":Jet_pt,"eta":Jet_eta,"phi":Jet_phi,"mass":Jet_mass})

nu_pz=np.load("nu_pz.npy")
nu_pt = get("MET_pt",numpy=True)
nu_phi = get("MET_phi",numpy=True)
nu_eta=np.arcsinh(nu_pz/nu_pt)

nu_4Vect=vector.awk({"pt":nu_pt,"eta":nu_eta,"phi":nu_phi,"mass":0*np.ones_like(nu_pt)})


mu_pt = get("Muon_pt[:,0]",numpy=True)
mu_eta= get("Muon_eta[:,0]",numpy=True)
mu_phi= get("Muon_phi[:,0]",numpy=True)
mu_4Vect=vector.awk({"pt":mu_pt,"eta":mu_eta,"phi":mu_phi,"mass":0.105*np.ones_like(mu_pt)})


btag=get("Jet_btagDeepB")
# %% deltaR

arange=list(np.arange(len(Jet_pt)))
nearest_jet_to_leptB_mask = ak.argmin(
    Jet_4Vect.deltaR(LeptB_LHE_4Vect), axis=1).to_list()
#nearest_jet_to_leptB_mask = [[elem] for elem in nearest_jet_to_leptB_mask]


deltaRmin_jet_leptB = Jet_4Vect.deltaR(LeptB_LHE_4Vect)[arange,nearest_jet_to_leptB_mask]

# I don't know why but the mask return and array instead of aMomentumRecord4D.The second line is a workaround to recast it
bJet_4Vect = Jet_4Vect[arange,nearest_jet_to_leptB_mask]
bJet_4Vect = vector.awk({"pt":bJet_4Vect.rho,"eta":bJet_4Vect.eta,"phi":bJet_4Vect.phi,"mass":bJet_4Vect.tau})


def remove(lista, elem):
    lista.remove(elem)
    return lista


others_Jet_mask = [remove(list(range(len(event))), bJetIdx)
              for event, bJetIdx in zip(Jet_phi, nearest_jet_to_leptB_mask)]

otherJet_4Vect = Jet_4Vect[others_Jet_mask]

# %% Rmin
h=Histogrammer(xlabel="$\Delta R_{min}$",bins=100,histrange=(0,1),ylim=(0,8000),legend_fontsize=20)
h.add_hist(deltaRmin_jet_leptB, label="$\Delta R_{min}$ jets-$b_{LHE}$(leptonic)",
           color=mcolors.XKCD_COLORS["xkcd:golden yellow"],edgecolor="black",linewidth=1.5)
h.plot()


# %%

Tmass_good=(bJet_4Vect+nu_4Vect+mu_4Vect).mass
Tmass_bad=ak.flatten((otherJet_4Vect+nu_4Vect+mu_4Vect).mass)

h=Histogrammer(xlabel="$m_{top}$ [GeV]",bins=100,histrange=(80,700),legend_fontsize=20,ylim=(0,20000))
h.add_hist(Tmass_bad, label="Other Jets",color="dodgerblue",edgecolor="black",linewidth=1.5)
h.add_hist(Tmass_good, label="Bjet (leptonic)",alpha=1,color=mcolors.XKCD_COLORS["xkcd:golden yellow"],edgecolor="black",linewidth=1.5)
h.plot()

#%%btag

btag_good=btag[arange,nearest_jet_to_leptB_mask]
btag_bad=ak.flatten(btag[others_Jet_mask])

h = Histogrammer(xlabel="btagDeepB", bins=100, histrange=(0, 1),legend_fontsize=20, ylim=(0, 20),density=True,ylabel="Density")
h.add_hist(btag_bad, label="Other Jets",color="dodgerblue",edgecolor="black",linewidth=2.5)

h.add_hist(btag_good, label="B jet (leptonic)",color=mcolors.XKCD_COLORS["xkcd:golden yellow"], edgecolor="black", linewidth=1.5,alpha=0.6)

h.plot()
plt.xlim(-0.03,1.03)