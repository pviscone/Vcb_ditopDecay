#%% Imports
import sys

sys.path.append("../../../utils")
import uproot
import ROOT
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from histogrammer import Histogrammer
import mplhep as hep
import matplotlib as mpl
import matplotlib.colors as mcolors


plt.style.use(hep.style.CMS)
signal = uproot.open(
    "../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root")["Events"]


def get(key, numpy=False, library="ak"):
    arr = signal.arrays(key, library=library)[key]
    if numpy == True:
        return arr.to_numpy()
    else:
        return arr

    


#%%
pdgId_Wdecay= get("LHEPart_pdgId")[:,[3,6]].to_numpy()

LHEmask = np.bitwise_or(pdgId_Wdecay ==13, pdgId_Wdecay == -13)

LeptB_LHE_phi= get("LHEPart_phi")[:,[2,5]].to_numpy()[LHEmask]
LeptB_LHE_eta= get("LHEPart_eta")[:,[2,5]].to_numpy()[LHEmask]
LeptB_LHE_pt= get("LHEPart_pt")[:,[2,5]].to_numpy()[LHEmask]

Jet_pt = get("Jet_pt")
Jet_eta = get("Jet_eta")
Jet_phi = get("Jet_phi")

nu_pz=np.load("nu_pz.npy")
nu_pt = get("MET_pt",numpy=True)
nu_phi = get("MET_phi",numpy=True)
nu_eta=np.arcsinh(nu_pz/nu_pt)

# %%
