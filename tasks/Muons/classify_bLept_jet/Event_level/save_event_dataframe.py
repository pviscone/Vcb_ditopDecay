#%%
#!-------------------Imports and load data-------------------!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from matplotlib.patches import Circle
from utils import np_and, padded_matrix, column, build_matrix, alternate_column, pad_and_alternate, circle,plot_events

import sys
sys.path.append("../../../../utils/")
from histogrammer import Histogrammer,xkcd_yellow




events = NanoEventsFactory.from_root(
    "../../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

nu_pz = np.load("../../neutrino_reco/nu_pz.npy")
nu_pt = events.MET.pt.to_numpy()
nu_phi = events.MET.phi.to_numpy()
nu_eta = np.arcsinh(nu_pz/nu_pt)

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


#%%
#!-------------------Selecting the objects-------------------!
#* LHE Mask
#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)
hadronic_LHE_mask=np.bitwise_not(leptonic_LHE_mask)

#*Define LHE objects
bLept_LHE = events.LHEPart[:, [2, 5]][leptonic_LHE_mask]
bHad_LHE = events.LHEPart[:, [2, 5]][hadronic_LHE_mask]
Wb_LHE = events.LHEPart[:, [4, 6]][hadronic_LHE_mask]
Wc_LHE = events.LHEPart[:, [3, 7]][hadronic_LHE_mask]

#*Define Jet objects
near_Jet,deltaR = events.LHEPart.nearest(events.Jet,return_metric=True)
bLept_Jet = near_Jet[:, [2, 5]][leptonic_LHE_mask]
bHad_Jet = near_Jet[:, [2, 5]][hadronic_LHE_mask]
Wb_Jet= near_Jet[:, [4, 6]][hadronic_LHE_mask]
Wc_Jet= near_Jet[:, [3, 7]][hadronic_LHE_mask]


#*Compute the deltaR between the LHE and the Jet
deltaRLept=deltaR[:, [2, 5]][leptonic_LHE_mask]
deltaRHad=deltaR[:, [2, 5]][hadronic_LHE_mask]
deltaRWb=deltaR[:, [4, 6]][hadronic_LHE_mask]
deltaRWc=deltaR[:, [3, 7]][hadronic_LHE_mask]

#*Compute the index in pt of the selected jets
bLept_pt_order = ak.argmin(events.Jet.delta_r(bLept_Jet), axis=1)
bHad_pt_order = ak.argmin(events.Jet.delta_r(bHad_Jet), axis=1)
Wb_pt_order = ak.argmin(events.Jet.delta_r(Wb_Jet), axis=1)
Wc_pt_order = ak.argmin(events.Jet.delta_r(Wc_Jet), axis=1)

n_max = 15
h=Histogrammer(histrange=(0,n_max),bins=n_max,histtype="step",xlabel="Jet Order in pt",ylabel="Events",log="y",ylim=(0.1,1e7))
h.add_hist(bLept_pt_order,label="bLept_pt_order")
h.add_hist(bHad_pt_order,label="bHad_pt_order")
h.add_hist(Wb_pt_order,label="Wb_pt_order")
h.add_hist(Wc_pt_order,label="Wc_pt_order")
h.plot()

#!-------------------delta R-------------------!
# * Apply the mask
Rmask = np_and(deltaRLept < 0.4,
              deltaRHad < 0.4,
              deltaRWb < 0.4,
              deltaRWc < 0.4,
              )

#%%
#!-------------------Multiple matching-------------------!



#*Compute the order of the jets in pt and compute the efficiency of selecting the first N

same_match_matrix = np.array([
    (bLept_pt_order == bHad_pt_order),
    (bLept_pt_order == Wb_pt_order),
    (bLept_pt_order == Wc_pt_order),
    (bHad_pt_order == Wb_pt_order),
    (bHad_pt_order == Wc_pt_order),
    (Wb_pt_order == Wc_pt_order)]).T



lab=["bLept==bHad","bLept==Wb","bLept==Wc","bHad==Wb","bHad==Wc","Wb==Wc"]
plt.barh(lab,width=np.sum(same_match_matrix[Rmask],axis=0)/np.sum(Rmask))



same_match_event_mask = np.bitwise_or.reduce(same_match_matrix, axis=1)


print(f"Fraction of event with at least 2 jet matching to the same parton: {np.sum(same_match_event_mask[Rmask])/np.sum(Rmask):.2%}")

# %%

#?Plot all the event which have multiple parton matching to the same jet
events.GenJet=events.GenJet[events.GenJet.pt>20]

label_list = ["bLept", "bHad", "Wc", "Wb"]
Jet_masked=events.Jet[Rmask]
GenJet_masked=events.GenJet[Rmask]
same_match_idx = np.where(same_match_event_mask[Rmask])[0]
LHE_list = [bLept_LHE[Rmask], bHad_LHE[Rmask], Wc_LHE[Rmask], Wb_LHE[Rmask]]
jet_list = [bLept_Jet[Rmask], bHad_Jet[Rmask], Wc_Jet[Rmask], Wb_Jet[Rmask]]

plot_events(
    LHE_list=LHE_list,
    jet_list=jet_list,
    label_list=label_list,
    Jets=Jet_masked,
    GenJets=GenJet_masked,
    index_list=same_match_idx[:10],
)


# %%
#!Inspect why the Wc match more often with the bLept than with the bHad
dR_bLept_Wc = bLept_LHE.delta_r(Wc_LHE)[Rmask]
dR_bHad_Wc = bHad_LHE.delta_r(Wc_LHE)[Rmask]
dR_bLept_Wb = bLept_LHE.delta_r(Wb_LHE)[Rmask]
dR_bHad_Wb = bHad_LHE.delta_r(Wb_LHE)[Rmask]


plt.figure(figsize=(16,10))
h = Histogrammer(bins=50, histrange=(0, 4),alpha=0.7,xlabel="$\Delta$R",ylabel="")
plt.subplot(1, 2, 1)
h.add_hist(dR_bLept_Wc, label="$\Delta$R bLept-$W_c$",color=xkcd_yellow,edgecolor="black")
h.add_hist(dR_bHad_Wc, label="$\Delta$R bHad-$W_c$",color="dodgerblue",edgecolor="blue")
h.plot()
plt.subplot(1, 2, 2)
h.add_hist(dR_bLept_Wb, label="$\Delta$R bLept-$W_b$",
           color=xkcd_yellow, edgecolor="black")
h.add_hist(dR_bHad_Wb, label="$\Delta$R bHad-$W_b$",
           color="dodgerblue", edgecolor="blue")
h.plot()


print(f"Fraction of event with dR bLept-Wc<bHad-Wc {ak.sum(dR_bLept_Wc<dR_bHad_Wc)/len(dR_bLept_Wc)}")
print(f"Fraction of event with dR bLept-Wc<0.4 {len(dR_bLept_Wc[dR_bLept_Wc<0.4])/len(dR_bLept_Wc)}")
print(f"Fraction of event with dR bHad-Wc<0.4 {len(dR_bHad_Wc[dR_bHad_Wc<0.4])/len(dR_bHad_Wc)}")
print(f"mean dR bLept-Wc {ak.mean(dR_bLept_Wc)}")
print(f"mean dR bHad-Wc {ak.mean(dR_bHad_Wc)}")
print(f"mean dR bLept-Wc dR<0.4 {ak.mean(dR_bLept_Wc[dR_bLept_Wc<0.4])}")
print(f"mean dR bHad-Wc dR<0.4 {ak.mean(dR_bHad_Wc[dR_bHad_Wc<0.4])}")


# %%
#!-------------------Masking-------------------!
num_jet_to_select = 6

R_multiple_match_mask = np.bitwise_and(Rmask, np.bitwise_not(same_match_event_mask))

mask = np_and(Rmask,
              np.bitwise_not(same_match_event_mask),
              bLept_pt_order < num_jet_to_select,
              bHad_pt_order < num_jet_to_select,
              Wb_pt_order < num_jet_to_select,
              Wc_pt_order < num_jet_to_select,)



efficiency = ak.sum(mask)/ak.sum(R_multiple_match_mask)
print(
    f"Percentage of events with selected jets in the first {num_jet_to_select} jets:{efficiency*100:.2f}%")




# %%
#!-------------------------SAVE-------------------------

# *Mask the objects
events.Jet = events.Jet[:, :num_jet_to_select]
events = events[mask]
events.Neutrino = nu_4Vect[mask]

bLept_Jet = bLept_Jet[mask]
bHad_Jet = bHad_Jet[mask]
Wb_Jet = Wb_Jet[mask]
Wc_Jet = Wc_Jet[mask]

# *Compute some invariant masses
events.Muon = events.Muon[:, 0]
events.Neutrino.Wmass = (events.Muon+events.Neutrino).mass
events.Tmass = (events.Neutrino+events.Muon+events.Jet).mass


col_labels = []

muon_matrix, mu_labels = build_matrix(events,"Muon", ["pt", "eta", "phi", "mass"])
nu_matrix, nu_labels = build_matrix(events,"Neutrino", ["pt", "eta", "phi", "mass", "Wmass"])
jet_matrix, jet_labels = pad_and_alternate(events,"Jet",
        ["pt", "eta", "phi", "mass", "btagDeepFlavB", "btagDeepFlavCvB", "btagDeepFlavCvL", "area", "Tmass"],
        num_jet_to_select)

bLept_label = column(bLept_pt_order[mask])
bHad_label = column(bHad_pt_order[mask])
Wb_label = column(Wb_pt_order[mask])
Wc_label = column(Wc_pt_order[mask])
col_labels = mu_labels+nu_labels+jet_labels + \
    ["bLept_label", "bHad_label", "Wb_label", "Wc_label"]

event_matrix = np.hstack(
    [muon_matrix, nu_matrix, jet_matrix, bLept_label, bHad_label, Wb_label, Wc_label])

event_df = pd.DataFrame(event_matrix, columns=col_labels)
event_df.to_pickle("./event_df.pkl", compression="bz2")

# %%
