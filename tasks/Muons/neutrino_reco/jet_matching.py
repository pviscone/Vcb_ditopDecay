
#%% Imports
import pandas as pd
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

# Return the minAbs or maxAbs of delta phi or eta between all the jets in the event
def minmax_delta_jets(jet_array, min_or_max, metric=None,level="event",flatten_axis=-1,numpy=True):
    if min_or_max not in ["min", "max"]:
        raise ValueError("min_or_max must be 'min' or 'max'")

    if metric == "phi":
        delta_matrix = jet_array[:, :, None].delta_phi(jet_array[:, None, :])
    elif metric == "eta":
        delta_matrix = jet_array[:, :, None].eta-jet_array[:, None, :].eta
    else:
        raise ValueError("metric must be 'phi' or 'eta'")

    delta_matrix = delta_matrix[delta_matrix != 0]

    if level=="event":
        delta_matrix = ak.flatten(delta_matrix, axis=2)
        argaxis=1
    elif level=="jet":
        argaxis=2
    else:
        raise ValueError("level must be 'event' or 'jet'")
    # Returns a mask of the minimum and maximum values.
    if min_or_max == "min":
        arg_mask = ak.argmin(np.abs(delta_matrix),
                             axis=argaxis,
                             keepdims=True)
    elif min_or_max == "max":
        arg_mask = ak.argmax(np.abs(delta_matrix),
                            axis=argaxis,
                            keepdims=True)

    res = delta_matrix[arg_mask]
    
    if flatten_axis==-1:
        res = ak.flatten(res, axis=None)
    else:
        res=ak.flatten(res,axis=flatten_axis)
    if numpy:
        res=res.to_numpy(allow_missing=False)
    return res


#%%



#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)

#Select only the LHE b from the top decay and then select the leptonic one
LeptB_LHE_4Vect=events.LHEPart[:,[2,5]][leptonic_LHE_mask]
mu_LHE_4Vect=events.LHEPart[:,[3,6]][leptonic_LHE_mask]

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
#%% dR muon LHE
deltaR_muons = mu_4Vect.delta_r(mu_LHE_4Vect)

h=Histogrammer(xlabel="$\Delta R$",bins=100,histrange=(0,1),ylim=(0,100000),legend_fontsize=20,log="y")
h.add_hist(deltaR_muons, label="$\Delta R$ $ \mu^{leading}-\mu_{LHE}$",
           color=xkcd_yellow,edgecolor="black",linewidth=2.5)
h.plot()
plt.plot([0.4,0.4],[0,100000],color="red",linestyle="-")
N_tot_muons=len(deltaR_muons)
deltaRMu_mask=deltaR_muons < 0.4
N_04_muons=np.sum(deltaRMu_mask)
plt.text(0.63,6000,f"$N_{{tot}}$   = {N_tot_muons}\n$N_{{<0.4}}=$ {N_04_muons} ({(N_04_muons/N_tot_muons):.2f})",fontsize=20)
plt.savefig("images/deltaR_Mu_LHE.png",bbox_inches='tight')
# %% Rmin

deltaRmin_jet_leptB = bJet_4Vect.delta_r(LeptB_LHE_4Vect)

h=Histogrammer(xlabel="$\Delta R_{min}$",bins=100,histrange=(0,1),ylim=(0,9000),legend_fontsize=20)
h.add_hist(deltaRmin_jet_leptB, label="$\Delta R_{min}$ jets-$b_{LHE}^{Lept}$",
           color=xkcd_yellow,edgecolor="black",linewidth=2.5)
h.plot()
plt.plot([0.4,0.4],[0,9000],color="red",linestyle="-")
N_tot=len(deltaRmin_jet_leptB)

deltaR_mask=deltaRmin_jet_leptB < 0.4
bJet_4Vect_leq04=bJet_4Vect[deltaR_mask]
nu_4Vect_leq04=nu_4Vect[deltaR_mask]
mu_4Vect_leq04=mu_4Vect[deltaR_mask]


N_04=np.sum(deltaR_mask)
plt.text(0.63,6000,f"$N_{{tot}}$   = {N_tot}\n$N_{{<0.4}}=$ {N_04} ({(N_04/N_tot):.2f})",fontsize=20)
plt.savefig("images/deltaRmin_jet_leptB.png",bbox_inches='tight')



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
plt.savefig("images/Tmass_jets.png",bbox_inches='tight')

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
plt.savefig("images/btag_jets.png",bbox_inches='tight')


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
plt.savefig("images/CvBtag_jets.png",bbox_inches='tight')

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
plt.savefig("images/CvLtag_jets.png",bbox_inches='tight')

# %% dphi mu-jet

h = Histogrammer(xlabel="$\Delta \phi$", bins=100, histrange=(-3.14,3.14), legend_fontsize=20, ylim=(0, 0.4), density=True, ylabel="Density", N=True, score=(0.3, 0.27))

h.add_hist(mu_4Vect_leq04.delta_phi(bJet_4Vect_leq04), label="$\mu-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue",edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(mu_4Vect.delta_phi(otherJet_4Vect)), label="$\mu-$Other Jets", color=xkcd_yellow,alpha=0.7,edgecolor="goldenrod",linewidth=2.5)
h.plot()
plt.savefig("images/dphi_mu_jets.png",bbox_inches='tight')

#%%deta mu-jet
h = Histogrammer(xlabel="$\Delta \eta$", bins=100, histrange=(-6,6), legend_fontsize=20,
                 ylim=(0, 0.6), density=True, ylabel="Density", N=True, score=(0.55, 0.41))

h.add_hist(mu_4Vect_leq04.eta-bJet_4Vect_leq04.eta,
           label="$\mu-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(mu_4Vect.eta-otherJet_4Vect.eta), label="$\mu-$Other Jets",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/deta_mu_jets.png",bbox_inches='tight')

# %%dphi met-jet
h = Histogrammer(xlabel="$\Delta \phi$", bins=100, histrange=(-3.14, 3.14), legend_fontsize=20,
                 ylim=(0, 0.4), density=True, ylabel="Density", N=True, score=(0.3, 0.27))

h.add_hist(events.MET[deltaR_mask].delta_phi(bJet_4Vect_leq04),
           label="$MET-Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)
h.add_hist(ak.flatten(events.MET.delta_phi(otherJet_4Vect)), label="$MET-$Other Jets",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("images/dphi_MET_jets.png",bbox_inches='tight')

#%% min dphi
otherJet_4Vect_leq04=otherJet_4Vect[deltaR_mask]



b_other_argminabs_deltaphi=ak.argmin(np.abs(bJet_4Vect_leq04.delta_phi(otherJet_4Vect_leq04)),axis=1).to_numpy(allow_missing=False)

arange=np.arange(len(bJet_4Vect_leq04))

b_other_min_deltaphi = bJet_4Vect_leq04.delta_phi(otherJet_4Vect_leq04)[arange,b_other_argminabs_deltaphi]

other_other_min_deltaphi = minmax_delta_jets(
    otherJet_4Vect_leq04, "min", metric="phi")


#---


is_min_deltaphi_from_b = np.array(ak.argmin(
    [np.abs(other_other_min_deltaphi),np.abs( b_other_min_deltaphi)], axis=0), dtype=bool)

min_deltaphi_from_b = b_other_min_deltaphi[is_min_deltaphi_from_b]
min_deltaphi_from_others = other_other_min_deltaphi[~is_min_deltaphi_from_b]


h = Histogrammer(xlabel="$\Delta \phi_{MinAbs}$", bins=100, histrange=(-2, 2), legend_fontsize=20,
                 ylim=(0, 2.2), density=True, ylabel="Density", N=True,score=(0.4, 1.4))

h.add_hist(min_deltaphi_from_b,
           label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$-Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(min_deltaphi_from_others, label="Others-Others",color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("./images/min_deltaphi_jets.png",bbox_inches='tight')

# %% maxdphi

b_other_argmaxabs_deltaphi = ak.argmax(np.abs(bJet_4Vect_leq04.delta_phi(
    otherJet_4Vect_leq04)), axis=1).to_numpy(allow_missing=False)

arange = np.arange(len(bJet_4Vect_leq04))

b_other_max_deltaphi = bJet_4Vect_leq04.delta_phi(
    otherJet_4Vect_leq04)[arange, b_other_argmaxabs_deltaphi]

other_other_max_deltaphi = minmax_delta_jets(
    otherJet_4Vect_leq04, "max", metric="phi")


# ---


is_max_deltaphi_from_b = np.array(ak.argmax(
    [np.abs(other_other_max_deltaphi), np.abs(b_other_max_deltaphi)], axis=0), dtype=bool)

max_deltaphi_from_b = b_other_max_deltaphi[is_max_deltaphi_from_b]
max_deltaphi_from_others = other_other_max_deltaphi[~is_max_deltaphi_from_b]


h = Histogrammer(xlabel="$\Delta \phi_{MaxAbs}$", bins=100, histrange=(-3.14, 3.14), legend_fontsize=20,
                 ylim=(0, 3), density=True, ylabel="Density", N=True, score=(0, 2))

h.add_hist(max_deltaphi_from_b,
           label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$-Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(max_deltaphi_from_others, label="Others-Others",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("./images/max_deltaphi_jets.png",bbox_inches='tight')

#%%maxeta jets


b_other_argmaxabs_deltaeta = ak.argmax(np.abs(bJet_4Vect_leq04.eta-(
    otherJet_4Vect_leq04.eta)), axis=1).to_numpy(allow_missing=False)

arange = np.arange(len(bJet_4Vect_leq04))

b_other_max_deltaeta = bJet_4Vect_leq04.eta-(otherJet_4Vect_leq04.eta)[arange, b_other_argmaxabs_deltaeta]

other_other_max_deltaeta= minmax_delta_jets(otherJet_4Vect_leq04,"max",metric="eta")

# ---


is_max_deltaeta_from_b = np.array(ak.argmax(
    [np.abs(other_other_max_deltaeta), np.abs(b_other_max_deltaeta)], axis=0), dtype=bool)

max_deltaeta_from_b = b_other_max_deltaeta[is_max_deltaeta_from_b]
max_deltaeta_from_others = other_other_max_deltaeta[~is_max_deltaeta_from_b]


h = Histogrammer(xlabel="$\Delta \eta_{MaxAbs}$", bins=100, histrange=(-9,9), legend_fontsize=20,
                 ylim=(0, 0.3), density=True, ylabel="Density", N=True, score=(0.05, 0.2))

h.add_hist(max_deltaeta_from_b,
           label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$-Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(max_deltaeta_from_others, label="Others-Others",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("./images/max_deltaeta_jets.png",bbox_inches='tight')

# %%mineta jets


b_other_argminabs_deltaeta = ak.argmin(np.abs(bJet_4Vect_leq04.eta-(
    otherJet_4Vect_leq04.eta)), axis=1).to_numpy(allow_missing=False)

arange = np.arange(len(bJet_4Vect_leq04))

b_other_min_deltaeta = bJet_4Vect_leq04.eta - \
    (otherJet_4Vect_leq04.eta)[arange, b_other_argminabs_deltaeta]
    
other_other_min_deltaeta = minmax_delta_jets(otherJet_4Vect_leq04,"min",metric="eta")

# ---


is_min_deltaeta_from_b = np.array(ak.argmin(
    [np.abs(other_other_min_deltaeta), np.abs(b_other_min_deltaeta)], axis=0), dtype=bool)

min_deltaeta_from_b = b_other_min_deltaeta[is_min_deltaeta_from_b]
min_deltaeta_from_others = other_other_min_deltaeta[~is_min_deltaeta_from_b]


h = Histogrammer(xlabel="$\Delta \eta_{MinAbs}$", bins=100, histrange=(-2,2), legend_fontsize=20,
                 ylim=(0, 3.5), density=True, ylabel="Density", N=True, score=(0.5, 2.2))

h.add_hist(min_deltaeta_from_b,
           label="$Bjet_{Lept}^{\Delta R_{LHE}\leq 0.4}$-Others", color="dodgerblue", edgecolor="black", linewidth=1.5, alpha=1)

h.add_hist(min_deltaeta_from_others, label="Others-Others",
           color=xkcd_yellow, alpha=0.7, edgecolor="goldenrod", linewidth=2.5)
h.plot()
plt.savefig("./images/min_deltaeta_jets.png",bbox_inches='tight')

#%% Save the features in a numpy matrix

df=pd.DataFrame()
#Save only the events in which the leptonic b jet is distant less than 0.4 from the LHE leptonic b jet
Jet_pt=events.Jet.pt[deltaR_mask]
Jet_eta=events.Jet.eta[deltaR_mask]
Jet_phi=events.Jet.phi[deltaR_mask]
Jet_mass = events.Jet.mass[deltaR_mask]
Jet_btag=events.Jet.btagDeepFlavB[deltaR_mask]
Jet_CvBtag=events.Jet.btagDeepFlavCvB[deltaR_mask]
Jet_CvLtag=events.Jet.btagDeepFlavCvL[deltaR_mask]
dPhi_Jet_mu=events.Jet.delta_phi(events.Muon[:,0])[deltaR_mask]
dPhi_Jet_nu=events.Jet.delta_phi(events.MET)[deltaR_mask]
dEta_Jet_mu=(events.Jet.eta-events.Muon[:,0].eta)[deltaR_mask]
dEta_Jet_nu=(events.Jet.eta-nu_4Vect.eta)[deltaR_mask]

T_mass=(events.Jet+events.Muon[:,0]+nu_4Vect)[deltaR_mask].mass

#min/max dphi deta jets
min_dPhi_Jets=minmax_delta_jets(events.Jet[deltaR_mask], "min",metric="phi",level="jet",flatten_axis=2,numpy=False)
max_dPhi_Jets=minmax_delta_jets(events.Jet[deltaR_mask], "max",metric="phi",level="jet",flatten_axis=2,numpy=False)

min_dEta_Jets=minmax_delta_jets(events.Jet[deltaR_mask], "min",metric="eta",level="jet",flatten_axis=2,numpy=False)
max_dEta_Jets=minmax_delta_jets(events.Jet[deltaR_mask], "max",metric="eta",level="jet",flatten_axis=2,numpy=False)


#Event id
#nJets_per_event = ak.count(events.Jet.pt, axis=-1)
#event_id=np.repeat(np.arange(len(events.Jet.pt))[deltaR_mask], nJets_per_event[deltaR_mask])

event_id=ak.Array(np.arange(len(events.Jet.pt))[deltaR_mask])



#LABEL: 1=Jet from Leptonic T, 0=Others

#[1]: return the metric (min delta_r)
#[:,[2,5]][leptonic_LHE_mask]: Select the leptonic b jet from the LHE
#[deltaR_mask]: Select only the events in which the leptonic b jet is distant less than 0.4 from the LHE jet
deltaR_jet_LHE=events.LHEPart.nearest(events.Jet,return_metric=True)[1][:, [2, 5]][leptonic_LHE_mask][deltaR_mask]

label = (events.Jet.delta_r(LeptB_LHE_4Vect)[
         deltaR_mask] == deltaR_jet_LHE)
label = ak.values_astype(label, int)

#Se usi una graph mettici anche il phi dei jet, all'interno dell' evento può impararci qualcosa sopra
features = ak.zip(
    {
    "Jet_pt":Jet_pt,
    "Jet_eta":Jet_eta,
    "Jet_phi":Jet_phi,
    "Jet_mass":Jet_mass,
    "Jet_btag": Jet_btag,
    "Jet_CvBtag": Jet_CvBtag,
    "Jet_CvLtag":Jet_CvLtag,
    "dPhi_Jet_mu":dPhi_Jet_mu,
    "dPhi_Jet_nu":dPhi_Jet_nu,
    "dEta_Jet_mu":dEta_Jet_mu,
    "dEta_Jet_nu":dEta_Jet_nu,
    "T_mass":T_mass,
    "min_dPhi_Jets":min_dPhi_Jets,
    "max_dPhi_Jets": max_dPhi_Jets,
    "min_dEta_Jets":min_dEta_Jets,
    "max_dEta_Jets":max_dEta_Jets,
    "label":label,
    "event_id":event_id
    })


#ak.to_parquet(features, "../classify_bLept_jet/Jet_features.pq", compression="lz4")

features_df = ak.to_pandas(features)
new_index = pd.MultiIndex.from_product(features_df.index.levels)

features_df = features_df.reindex(new_index)
perm1=np.random.permutation(len(Jet_pt))
perm2=np.random.permutation(15)
i, j = features_df.index.levels
idx = pd.IndexSlice

features_df = features_df.loc[i[perm1]]
features_df = features_df.loc[idx[i, j[perm2]],:]
features_df=features_df.dropna()


features_df.to_pickle("../classify_bLept_jet/Jet_level/Jet_features.pkl",compression="bz2")
