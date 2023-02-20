
#%% Imports
import sys
sys.path.append("../../../utils")


import histogrammer
import importlib
importlib.reload(histogrammer)
Histogrammer = histogrammer.Histogrammer

import mplhep
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



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
           color=xkcd_yellow,edgecolor="black",linewidth=2.5)
h.plot()
plt.plot([0.4,0.4],[0,9000],color="red",linestyle="-")
N_tot=len(deltaRmin_jet_leptB)

deltaR_mask=deltaRmin_jet_leptB < 0.4

N_04=np.sum(deltaR_mask)
plt.text(0.63,6000,f"$N_{{tot}}$   = {N_tot}\n$N_{{<0.4}}=$ {N_04} ({(N_04/N_tot):.2f})",fontsize=20)
plt.savefig("images/deltaRmin_jet_leptB.png")

bJet_4Vect_leq04=bJet_4Vect[deltaR_mask]
nu_4Vect_leq04=nu_4Vect[deltaR_mask]
mu_4Vect_leq04=mu_4Vect[deltaR_mask]

# %%
# _good =right b jet (t->b(W->lv))
# _bad =all other jets
Tmass_good=(bJet_4Vect_leq04+nu_4Vect_leq04+mu_4Vect_leq04).mass
Tmass_bad=ak.flatten((otherJet_4Vect+nu_4Vect+mu_4Vect).mass)

h=Histogrammer(xlabel="$M_{top}$ [GeV]",bins=100,histrange=(80,700),legend_fontsize=22,density=True,ylim=(0,0.016),ylabel="Density",fontsize=30,N=True,score=(350,0.01))

h.add_hist(Tmass_good, label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$ + $W_{lept}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(Tmass_bad, label="Other Jets + $W_{lept}$", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.7)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("images/Tmass_jets.png")

#%%btag
# _good =right b jet (t->b(W->lv)) (+ <0.4 deltaR cut)
# _bad =all other jets
btag_good=bJet_4Vect_leq04.btagDeepFlavB.to_numpy()
btag_bad=ak.flatten(otherJet_4Vect.btagDeepFlavB).to_numpy()

h = Histogrammer(xlabel="btagDeepFlavB", bins=100, histrange=(0, 1),legend_fontsize=20, ylim=(0, 38),density=True,ylabel="Density",N=True,score=(0.61,25))

h.add_hist(btag_good, label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue",
           edgecolor="black", linewidth=2.5, alpha=1)

h.add_hist(btag_bad, label="Others",color=xkcd_yellow,alpha=0.7,edgecolor="goldenrod",linewidth=2.5)



h.plot()
plt.xlim(-0.03,1.03)
plt.savefig("images/btag_jets.png")


# %% ctag
# _good =right b jet (t->b(W->lv)) (+ <0.4 deltaR cut)
# _bad =all other jets
CvBtag_good = bJet_4Vect_leq04.btagDeepFlavCvB.to_numpy()
CvBtag_bad = ak.flatten(otherJet_4Vect.btagDeepFlavCvB).to_numpy()

h = Histogrammer(xlabel="btagDeepFlavCvB", bins=100, histrange=(
    0, 1), legend_fontsize=20, ylim=(0, 38), density=True, ylabel="Density", N=True,score=(0.6,25))

h.add_hist(CvBtag_good, label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue",
           edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(CvBtag_bad, label="Others", color=xkcd_yellow,
           alpha=0.7, edgecolor="goldenrod", linewidth=2.5)


h.plot()
plt.xlim(-0.03, 1.03)
plt.savefig("images/CvBtag_jets.png")

#%% CvLtag

# _good =right b jet (t->b(W->lv)) (+ <0.4 deltaR cut)
# _bad =all other jets
CvLtag_good = bJet_4Vect_leq04.btagDeepFlavCvL.to_numpy()
CvLtag_bad = ak.flatten(otherJet_4Vect.btagDeepFlavCvL).to_numpy()

h = Histogrammer(xlabel="btagDeepFlavCvL", bins=100, histrange=(
    0, 1), legend_fontsize=20, ylim=(0, 10), density=True, ylabel="Density", N=True,score=(0.6,6.5))

h.add_hist(CvLtag_good, label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue",
           edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(CvLtag_bad, label="Others", color=xkcd_yellow,
           alpha=0.7, edgecolor="goldenrod", linewidth=2.5)


h.plot()
plt.xlim(-0.03, 1.03)
plt.savefig("images/CvLtag_jets.png")

# %% dphi mu-jet

h = Histogrammer(xlabel="$\Delta \phi$", bins=100, histrange=(-3.14,3.14), legend_fontsize=20, ylim=(0, 0.4), density=True, ylabel="Density", N=True, score=(0.3, 0.27))

h.add_hist(mu_4Vect_leq04.delta_phi(bJet_4Vect_leq04), label="$\mu-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue",edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(mu_4Vect.delta_phi(otherJet_4Vect)), label="$\mu-$Other Jets", color=xkcd_yellow,alpha=0.7,edgecolor="goldenrod",linewidth=2.5)
h.plot()
plt.savefig("images/dphi_mu_jets.png")

#%%deta mu-jet
h = Histogrammer(xlabel="$\Delta \eta$", bins=100, histrange=(-6,6), legend_fontsize=20,
                 ylim=(0, 0.6), density=True, ylabel="Density", N=True, score=(0.55, 0.41))

h.add_hist(mu_4Vect_leq04.eta-bJet_4Vect_leq04.eta,
           label="$\mu-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(mu_4Vect.eta-otherJet_4Vect.eta), label="$\mu-$Other Jets",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/deta_mu_jets.png")

# %%dphi met-jet
h = Histogrammer(xlabel="$\Delta \phi$", bins=100, histrange=(-3.14, 3.14), legend_fontsize=20,
                 ylim=(0, 0.4), density=True, ylabel="Density", N=True, score=(0.3, 0.27))

h.add_hist(events.MET[deltaR_mask].delta_phi(bJet_4Vect_leq04),
           label="$MET-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(events.MET.delta_phi(otherJet_4Vect)), label="$MET-$Other Jets",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/dphi_MET_jets.png")

# %%
#min max delta phi and eta between bjet and others
minphi = ak.min(bJet_4Vect_leq04.delta_phi(otherJet_4Vect[deltaR_mask]),axis=1)
maxphi = ak.max(bJet_4Vect_leq04.delta_phi(otherJet_4Vect[deltaR_mask]),axis=1)
h = Histogrammer(xlabel="$\Delta \phi$", bins=100, histrange=(-3.14, 3.14), legend_fontsize=20,
                 ylim=(0, 1.7), density=True, ylabel="Density", score=(0.4, 1.1))

h.add_hist(minphi,
           label="Min $Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}-$Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(maxphi, label="Max $Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}-$Others",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/min_max_dphi_jets.png")

# %% min max delta eta between bjet and others

minphi = ak.min(bJet_4Vect_leq04.eta-(
    otherJet_4Vect[deltaR_mask].eta), axis=1)
maxphi = ak.max(bJet_4Vect_leq04.eta-(
    otherJet_4Vect[deltaR_mask].eta), axis=1)
h = Histogrammer(xlabel="$\Delta \eta$", bins=100, histrange=(-7,7), legend_fontsize=20,
                 ylim=(0, 0.4), density=True, ylabel="Density", score=(1, 0.27))

h.add_hist(minphi,
           label="Min $Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}-$Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(maxphi, label="Max $Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}-$Others",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/min_max_deta_jets.png")
