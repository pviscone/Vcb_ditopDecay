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
from selections import obj_selection,mu_selection


xkcd_yellow = mcolors.XKCD_COLORS["xkcd:golden yellow"]
mplhep.style.use(["CMS", "fira", "firamath"])


# I don't know why but the first plot (only) is always weird.
# So,as a workaround, these two lines create a dummy plot.
plt.hist([0])
plt.close()

events = NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal/signal_train.root",
    schemaclass=NanoAODSchema,
).events()

events=obj_selection(events)
events=mu_selection(events)

#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)

#Select only the LHE b from the top decay and then select the leptonic one

b_LHE_mask=np.array([[2,5]]*len(events))[leptonic_LHE_mask]
bLHE=events.LHEPart[np.arange(len(events)),b_LHE_mask]

mu_LHE_mask=b_LHE_mask=np.array([[3,6]]*len(events))[leptonic_LHE_mask]
mu_LHE_4Vect=events.LHEPart[np.arange(len(events)),mu_LHE_mask]

All_Jets_4Vect=events.Jet
b = events.LHEPart.nearest(All_Jets_4Vect)[np.arange(len(events)),b_LHE_mask]


#Select the other jets
otherJet_mask = (All_Jets_4Vect.delta_r(b) > 0.00001)
others = All_Jets_4Vect[otherJet_mask]

mu=events.Muon[:,0]
nu=events.MET
W=events.W

deltaRmin_jet_leptB = b.delta_r(bLHE)
deltaR_mask=deltaRmin_jet_leptB < 0.4
b=b[deltaR_mask]
nu=nu[deltaR_mask]
mu=mu[deltaR_mask]
W=W[deltaR_mask]
others=others[deltaR_mask]


nu = ak.zip(
    {
        "pt": nu.pt,
        "eta": nu.eta,
        "phi": nu.phi,
        "mass": np.zeros_like(nu.pt),
    },
    with_name="PtEtaPhiMLorentzVector",
    behavior=vector.behavior,
)

#%%
def stacked(list_plot,
            bins=None,
            label=None,
            ylabel="",
            xlabel="",
            units="",
            savefig=False,
            yfactor=1.2,
            log=False,
            colors=["dodgerblue",xkcd_yellow],):
    
    plt.rc("font", size=30)
    plt.rc('legend', fontsize=22)
    
    for i in range(len(list_plot)):
        label[i]=label[i]+f"\n{np.mean(list_plot[i]):.2f}({np.std(list_plot[i]):.2f}) {units}"
    
    h=plt.hist(list_plot,stacked=True, color=colors,label=label,bins=bins)
    plt.grid()
    plt.legend()
    plt.ylabel(ylabel)
    plt.ylim(0,np.max(h[0][-1])*yfactor)
    mplhep.cms.text("Private Work",loc=2)
    plt.xlabel(xlabel)
    plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    if log:
        plt.yscale("log")
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    return h


#%%
#!PT
h=Histogrammer(xlabel="$p_{T}$ [GeV]",
               bins=40,
               histrange=(20,300),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True)

h.add_hist(b.pt, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others.pt), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/Jet_pt.png",bbox_inches='tight')
