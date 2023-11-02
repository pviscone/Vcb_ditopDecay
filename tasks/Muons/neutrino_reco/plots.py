
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

# %%

#!TMASS
TMass_b=((b+nu+mu).mass)
TMass_others=((others+nu+mu).mass)

h=Histogrammer(xlabel="$M_{t}$ [GeV]",
               bins=100,
               histrange=(80,700),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(TMass_b, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(TMass_others), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.7)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/Tmass.png",bbox_inches='tight')


# %%
TMass_others_min=np.abs(ak.singletons(ak.min(TMass_others,axis=1))-173)
TMass_both=ak.concatenate([np.abs(ak.singletons(TMass_b)-173),TMass_others_min],axis=1)
TMass_argmin=ak.argmin(TMass_both,axis=1)

TMass_min_b=TMass_both[TMass_argmin==0][:,0]
TMass_min_others=TMass_both[TMass_argmin==1][:,1]


stacked([TMass_min_b,TMass_min_others],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,150,40),
        units="[GeV]",
        xlabel="$min(|M_{t}-173|)$ [GeV]",
        savefig="plots/min_Tmass.png"
        )


#%%
#!PT
h=Histogrammer(xlabel="$p_{T}$ [GeV]",
               bins=40,
               histrange=(20,300),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b.pt, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others.pt), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/Jet_pt.png",bbox_inches='tight')

#%%
#! Max pt

Pt_others_max=(ak.singletons(ak.max(others.pt,axis=1)))
Pt_both=ak.concatenate([(ak.singletons(b.pt)),Pt_others_max],axis=1)
Pt_argmax=ak.argmax(Pt_both,axis=1)

Pt_max_b=Pt_both[Pt_argmax==0][:,0]
Pt_max_others=Pt_both[Pt_argmax==1][:,1]


stacked([Pt_max_b,Pt_max_others],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","max(Others)"],
        bins=np.linspace(20,300,40),
        units="[GeV]",
        xlabel="$max(p_T)$ [GeV]",
        savefig="plots/max_Pt.png"
        )
#%%
#!B
h=Histogrammer(xlabel="$DeepJet B$",
               bins=40,
               histrange=(0,1),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b.btagDeepFlavB, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others.btagDeepFlavB), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/btag.png",bbox_inches='tight')
#%%
max_b=ak.max(others.btagDeepFlavB,axis=1)
b_b=b.btagDeepFlavB
both_b=ak.concatenate([ak.singletons(b_b),ak.singletons(max_b)],axis=1)

argmax_b=ak.argmax(both_b,axis=1)
stacked([both_b[argmax_b==0][:,0],both_b[argmax_b==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","max(Others)"],
        bins=np.linspace(0.9,1,40),
        units="",
        xlabel="$max(DeepJet B)$",
        yfactor=1.4,
        savefig="plots/max_btab.png"
        )

#%%
#!CVB


h=Histogrammer(xlabel="$DeepJet CvB$",
               bins=40,
               histrange=(0,1),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b.btagDeepFlavCvB, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others.btagDeepFlavCvB), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/cvb.png",bbox_inches='tight')
#%%
max_b=ak.min(others.btagDeepFlavCvB,axis=1)
b_b=b.btagDeepFlavCvB
both_b=ak.concatenate([ak.singletons(b_b),ak.singletons(max_b)],axis=1)

argmax_b=ak.argmin(both_b,axis=1)
stacked([both_b[argmax_b==0][:,0],both_b[argmax_b==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0.,0.3,40),
        units="",
        xlabel="$min(DeepJet B)$",
        yfactor=1.4,
        savefig="plots/min_cvb.png"
        )

#%%
#!CvL


h=Histogrammer(xlabel="$DeepJet CvL$",
               bins=40,
               histrange=(0,1),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b.btagDeepFlavCvL, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others.btagDeepFlavCvL), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/CvL.png",bbox_inches='tight')



# %%
#!delta phi-mu

b_mu_dphi=np.abs(b.delta_phi(mu))
others_mu_dphi=np.abs(others.delta_phi(mu))

h=Histogrammer(xlabel=r"$|\Delta \phi(j-\mu)|$",
               bins=40,
               histrange=(0,3.14),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_mu_dphi, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_mu_dphi), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dphi_mu.png",bbox_inches='tight')


#%%
min_others_mu_dphi=ak.min(others_mu_dphi,axis=1)
both_mu_dphi=ak.concatenate([ak.singletons(b_mu_dphi),ak.singletons(min_others_mu_dphi)],axis=1)

argmax_mu_dphi=ak.argmin(both_b,axis=1)

stacked([both_mu_dphi[argmax_mu_dphi==0][:,0],both_mu_dphi[argmax_mu_dphi==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3.14,40),
        units="",
        xlabel=r"min($|\Delta \phi(j-\mu)|$)",
        yfactor=1.4,
        savefig="plots/min_dphi_mu.png"
        )


#%%
#!

#!delta r-mu

b_mu_dr=np.abs(b.delta_r(mu))
others_mu_dr=np.abs(others.delta_r(mu))

h=Histogrammer(xlabel=r"$|\Delta R(j-\mu)|$",
               bins=40,
               histrange=(0,5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_mu_dr, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_mu_dr), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dr_mu.png",bbox_inches='tight')


#%%
min_others_mu_dr=ak.min(others_mu_dr,axis=1)
both_mu_dr=ak.concatenate([ak.singletons(b_mu_dr),ak.singletons(min_others_mu_dr)],axis=1)

argmax_mu_dr=ak.argmin(both_b,axis=1)

stacked([both_mu_dr[argmax_mu_dr==0][:,0],both_mu_dr[argmax_mu_dr==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,5,40),
        units="",
        xlabel=r"min($|\Delta R(j-\mu)|$)",
        yfactor=1.4,
        savefig="plots/min_dr_mu.png"
        )


#%%

#!delta eta-mu

b_mu_deta=np.abs(b.eta-mu.eta)
others_mu_deta=np.abs(others.eta-mu.eta)

h=Histogrammer(xlabel=r"$|\Delta \eta(j-\mu)|$",
               bins=40,
               histrange=(0,5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_mu_deta, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_mu_deta), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/deta_mu.png",bbox_inches='tight')


#%%
min_others_mu_deta=ak.min(others_mu_deta,axis=1)
both_mu_deta=ak.concatenate([ak.singletons(b_mu_deta),ak.singletons(min_others_mu_deta)],axis=1)

argmax_mu_deta=ak.argmin(both_b,axis=1)

stacked([both_mu_deta[argmax_mu_deta==0][:,0],both_mu_deta[argmax_mu_deta==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3,40),
        units="",
        xlabel=r"min($|\Delta \eta(j-\mu)|$)",
        yfactor=1.4,
        savefig="plots/min_deta_mu.png"
        )



#%%
#!delta phi-nu

b_nu_dphi=np.abs(b.delta_phi(nu))
others_nu_dphi=np.abs(others.delta_phi(nu))

h=Histogrammer(xlabel=r"$|\Delta \phi(j-\nu)|$",
               bins=40,
               histrange=(0,3.14),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_nu_dphi, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_nu_dphi), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dphi_nu.png",bbox_inches='tight')


#%%
min_others_nu_dphi=ak.min(others_nu_dphi,axis=1)
both_nu_dphi=ak.concatenate([ak.singletons(b_nu_dphi),ak.singletons(min_others_nu_dphi)],axis=1)

argmax_nu_dphi=ak.argmin(both_b,axis=1)

stacked([both_nu_dphi[argmax_nu_dphi==0][:,0],both_nu_dphi[argmax_nu_dphi==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3.14,40),
        units="",
        xlabel=r"min($|\Delta \phi(j-\nu)|$)",
        yfactor=1.4,
        savefig="plots/min_dphi_nu.png"
        )


#%%
#!

#!delta r-nu

b_nu_dr=np.abs(b.delta_r(nu))
others_nu_dr=np.abs(others.delta_r(nu))

h=Histogrammer(xlabel=r"$|\Delta R(j-\nu)|$",
               bins=40,
               histrange=(0,5.5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.6,0.65])

h.add_hist(b_nu_dr, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_nu_dr), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dr_nu.png",bbox_inches='tight')


#%%
min_others_nu_dr=ak.min(others_nu_dr,axis=1)
both_nu_dr=ak.concatenate([ak.singletons(b_nu_dr),ak.singletons(min_others_nu_dr)],axis=1)

argmax_nu_dr=ak.argmin(both_b,axis=1)

stacked([both_nu_dr[argmax_nu_dr==0][:,0],both_nu_dr[argmax_nu_dr==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,5,40),
        units="",
        xlabel=r"min($|\Delta R(j-\nu)|$)",
        yfactor=1.4,
        savefig="plots/min_dr_nu.png"
        )


#%%

#!delta eta-nu

b_nu_deta=np.abs(b.eta-nu.eta)
others_nu_deta=np.abs(others.eta-nu.eta)

h=Histogrammer(xlabel=r"$|\Delta \eta(j-\nu)|$",
               bins=40,
               histrange=(0,5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_nu_deta, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_nu_deta), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/deta_nu.png",bbox_inches='tight')


#%%
min_others_nu_deta=ak.min(others_nu_deta,axis=1)
both_nu_deta=ak.concatenate([ak.singletons(b_nu_deta),ak.singletons(min_others_nu_deta)],axis=1)

argmax_nu_deta=ak.argmin(both_b,axis=1)

stacked([both_nu_deta[argmax_nu_deta==0][:,0],both_nu_deta[argmax_nu_deta==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3,40),
        units="",
        xlabel=r"min($|\Delta \eta(j-\nu)|$)",
        yfactor=1.4,
        savefig="plots/min_deta_nu.png"
        )


#%%

#!delta phi-W

b_W_dphi=np.abs(b.delta_phi(W))
others_W_dphi=np.abs(others.delta_phi(W))

h=Histogrammer(xlabel=r"$|\Delta \phi(j-W)|$",
               bins=40,
               histrange=(0,3.14),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_W_dphi, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_W_dphi), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dphi_W.png",bbox_inches='tight')


#%%
min_others_W_dphi=ak.min(others_W_dphi,axis=1)
both_W_dphi=ak.concatenate([ak.singletons(b_W_dphi),ak.singletons(min_others_W_dphi)],axis=1)

argmax_W_dphi=ak.argmin(both_b,axis=1)

stacked([both_W_dphi[argmax_W_dphi==0][:,0],both_W_dphi[argmax_W_dphi==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3.14,40),
        units="",
        xlabel=r"min($|\Delta \phi(j-W)|$)",
        yfactor=1.4,
        savefig="plots/min_dphi_W.png"
        )


#%%
#!

#!delta r-W

b_W_dr=np.abs(b.delta_r(W))
others_W_dr=np.abs(others.delta_r(W))

h=Histogrammer(xlabel=r"$|\Delta R(j-W)|$",
               bins=40,
               histrange=(0,5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.67,0.65])

h.add_hist(b_W_dr, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_W_dr), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/dr_W.png",bbox_inches='tight')


#%%
min_others_W_dr=ak.min(others_W_dr,axis=1)
both_W_dr=ak.concatenate([ak.singletons(b_W_dr),ak.singletons(min_others_W_dr)],axis=1)

argmax_W_dr=ak.argmin(both_b,axis=1)

stacked([both_W_dr[argmax_W_dr==0][:,0],both_W_dr[argmax_W_dr==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,5,40),
        units="",
        xlabel=r"min($|\Delta R(j-W)|$)",
        yfactor=1.4,
        savefig="plots/min_dr_W.png"
        )


#%%

#!delta eta-W

b_W_deta=np.abs(b.eta-W.eta)
others_W_deta=np.abs(others.eta-W.eta)

h=Histogrammer(xlabel=r"$|\Delta \eta(j-W)|$",
               bins=40,
               histrange=(0,5),
               ylabel="Density",density=True,
               legend_fontsize=22,fontsize=30,
               grid=True,
               score=[0.53,0.65])

h.add_hist(b_W_deta, label=r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", alpha=1,
           color="dodgerblue", edgecolor="black", linewidth=3)

h.add_hist(ak.flatten(others_W_deta), label="Others", color=xkcd_yellow,edgecolor="goldenrod", linewidth=2.5,alpha=0.6)
h.plot()
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
plt.savefig("plots/deta_W.png",bbox_inches='tight')


#%%
min_others_W_deta=ak.min(others_W_deta,axis=1)
both_W_deta=ak.concatenate([ak.singletons(b_W_deta),ak.singletons(min_others_W_deta)],axis=1)

argmax_W_deta=ak.argmin(both_b,axis=1)

stacked([both_W_deta[argmax_W_deta==0][:,0],both_W_deta[argmax_W_deta==1][:,1]],
        label=[r"$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$","min(Others)"],
        bins=np.linspace(0,3,40),
        units="",
        xlabel=r"min($|\Delta \eta(j-W)|$)",
        yfactor=1.4,
        savefig="plots/min_deta_W.png"
        )
# %%
#!Corner
labels=[r"$m_t$",r"$p_T$","b-tag","CvB","CvL",r"$\Delta \phi (j-\mu)$",r"$\Delta \phi (j-\nu)$",r"$\Delta \phi (j-W)$",r"$\Delta \eta (j-\mu)$",r"$\Delta \eta (j-\nu)$",r"$\Delta \eta (j-W)$",r"$\Delta R (j-\mu)$",r"$\Delta R (j-\nu)$",r"$\Delta R (j-W)$"]
b_corner=np.stack([TMass_b,b.pt,b.btagDeepFlavB,b.btagDeepFlavCvB,b.btagDeepFlavCvL,b_mu_dphi,b_nu_dphi,b_W_dphi,b_mu_deta,b_nu_deta,b_W_deta,b_mu_dr,b_nu_dr,b_W_dr],axis=1).to_numpy()

others_corner=np.stack([ak.flatten(TMass_others),
                        ak.flatten(others.pt),
                        ak.flatten(others.btagDeepFlavB),
                        ak.flatten(others.btagDeepFlavCvB),
                        ak.flatten(others.btagDeepFlavCvL),
                        ak.flatten(others_mu_dphi),
                        ak.flatten(others_nu_dphi),
                        ak.flatten(others_W_dphi),
                        ak.flatten(others_mu_deta),
                        ak.flatten(others_nu_deta),
                        ak.flatten(others_W_deta),
                        ak.flatten(others_mu_dr),
                        ak.flatten(others_nu_dr),
                        ak.flatten(others_W_dr)],axis=1).to_numpy()

GTC = pygtc.plotGTC(chains=[b_corner,others_corner],
                    paramNames=labels,
                    chainLabels=["bLept","Others"])