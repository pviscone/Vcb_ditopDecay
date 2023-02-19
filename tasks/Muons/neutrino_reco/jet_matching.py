
#%% Imports
import mplhep
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys

sys.path.append("../../../utils")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from histogrammer import Histogrammer



xkcd_yellow = mcolors.XKCD_COLORS["xkcd:golden yellow"]
mplhep.style.use(["CMS", "fira", "firamath"])


# I don't know why but the first plot (only) is always weird.
# So,as a workaround, these two lines create a dummy plot.
plt.hist([0])
plt.close()

events = NanoEventsFactory.from_root(
    "../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

#%%



#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)

#Select only the LHE b from the top decay and then select the leptonic one
LeptB_LHE_4Vect=events.LHEPart[:,[2,5]][leptonic_LHE_mask]

All_Jets_4Vect=events.Jet
bJet_4Vect = events.LHEPart.nearest(All_Jets_4Vect)[:, [2, 5]][leptonic_LHE_mask]

#Select the other jets
otherJet_mask = (All_Jets_4Vect.delta_r(bJet_4Vect) > 0.00001)
otherJet_4Vect = All_Jets_4Vect[otherJet_mask]

nu_pz=np.load("nu_pz.npy")
nu_pt = events.MET.pt.to_numpy()
nu_phi = events.MET.phi.to_numpy()
nu_eta=np.arcsinh(nu_pz/nu_pt)

nu_4Vect = ak.zip(
    {
        "pt": nu_pt,
        "eta": nu_eta,
        "phi": nu_phi,
        "mass": np.zeros_like(nu_pt),
    },
    with_name="PtEtaPhiMLorentzVector",
    behavior=vector.behavior,
)
del nu_pt, nu_eta, nu_phi, nu_pz


mu_4Vect=events.Muon[:,0]


# %% Rmin

deltaRmin_jet_leptB = bJet_4Vect.delta_r(LeptB_LHE_4Vect)

h=Histogrammer(xlabel="$\Delta R_{min}$",bins=100,histrange=(0,1),ylim=(0,9000),legend_fontsize=20)
h.add_hist(deltaRmin_jet_leptB, label="$\Delta R_{min}$ jets-$b_{LHE}^{Lept}$",
           color=xkcd_yellow,edgecolor="black",linewidth=2)
h.plot()

plt.savefig("images/deltaRmin_jet_leptB.png")


# %%
# _good =right b jet (t->b(W->lv))
# _bad =all other jets
Tmass_good=(bJet_4Vect+nu_4Vect+mu_4Vect).mass
Tmass_bad=ak.flatten((otherJet_4Vect+nu_4Vect+mu_4Vect).mass)

h=Histogrammer(xlabel="$M_{top}$ [GeV]",bins=100,histrange=(80,700),legend_fontsize=22,density=True,ylim=(0,0.015),ylabel="Density",fontsize=30,N=True)

h.add_hist(Tmass_good, label="$Bjet_{lept}$ + $W_{lept}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=1.5)

h.add_hist(Tmass_bad, label="Other Jets + $W_{lept}$", color=xkcd_yellow,edgecolor="black", linewidth=1.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("images/Tmass_jets.png")

#%%btag
# _good =right b jet (t->b(W->lv))
# _bad =all other jets
btag_good=bJet_4Vect.btagDeepB.to_numpy()
btag_bad=ak.flatten(otherJet_4Vect.btagDeepB).to_numpy()

h = Histogrammer(xlabel="btagDeepB", bins=100, histrange=(0, 1),legend_fontsize=20, ylim=(0, 20),density=True,ylabel="Density",N=True)

h.add_hist(btag_good, label="B jet (leptonic)", color="dodgerblue",
           edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(btag_bad, label="Other Jets",color=xkcd_yellow,alpha=0.6,edgecolor="black",linewidth=2.5)



h.plot()
plt.xlim(-0.03,1.03)
plt.savefig("images/btag_jets.png")

#!Achtunh! There is a non negligible fraction of events with btagDeepB==-1. How should I interpret this?
# %%
