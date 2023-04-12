
#%% Imports
import pandas as pd
import sys
sys.path.append("../../../utils")



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


def np_and(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.bitwise_and(arg, res)
    return res

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


#filepath="../../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root"
#nupath="../../nu_pz.npy"
filepath="../../BigMuon_MuonSelection.root"
nupath="../../BigMuons_nu_pz.npy"


events = NanoEventsFactory.from_root(
    filepath,
    schemaclass=NanoAODSchema,
).events()

nu_pz = np.load(nupath)
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




# %% Rmin
#* LHE Mask
#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)
hadronic_LHE_mask=np.bitwise_not(leptonic_LHE_mask)

bLept_LHE = events.LHEPart[:, [2, 5]][leptonic_LHE_mask]

#*Compute the index in pt of the selected jets
near_Jet,deltaR = events.LHEPart.nearest(events.Jet,return_metric=True)
bLept_Jet = near_Jet[:, [2, 5]][leptonic_LHE_mask]
bHad_Jet = near_Jet[:, [2, 5]][hadronic_LHE_mask]
Wb_Jet= near_Jet[:, [4, 6]][hadronic_LHE_mask]
Wc_Jet= near_Jet[:, [3, 7]][hadronic_LHE_mask]

bLept_pt_order = ak.argmin(events.Jet.delta_r(bLept_Jet), axis=1)
bHad_pt_order = ak.argmin(events.Jet.delta_r(bHad_Jet), axis=1)
Wb_pt_order = ak.argmin(events.Jet.delta_r(Wb_Jet), axis=1)
Wc_pt_order = ak.argmin(events.Jet.delta_r(Wc_Jet), axis=1)

deltaRLept=deltaR[:, [2, 5]][leptonic_LHE_mask]
deltaRHad=deltaR[:, [2, 5]][hadronic_LHE_mask]
deltaRWb=deltaR[:, [4, 6]][hadronic_LHE_mask]
deltaRWc=deltaR[:, [3, 7]][hadronic_LHE_mask]

Rmask = np_and(deltaRLept < 0.4,
              deltaRHad < 0.4,
              deltaRWb < 0.4,
              deltaRWc < 0.4,
              )

same_match_matrix = np.array([
    (bLept_pt_order == bHad_pt_order),
    (bLept_pt_order == Wb_pt_order),
    (bLept_pt_order == Wc_pt_order),
    (bHad_pt_order == Wb_pt_order),
    (bHad_pt_order == Wc_pt_order),
    (Wb_pt_order == Wc_pt_order)]).T
same_match_event_mask = np.bitwise_or.reduce(same_match_matrix, axis=1)


num_jet_to_select = 7

R_multiple_match_mask = np.bitwise_and(Rmask, np.bitwise_not(same_match_event_mask))

mask = np_and(Rmask,
              np.bitwise_not(same_match_event_mask),
              bLept_pt_order < num_jet_to_select,
              bHad_pt_order < num_jet_to_select,
              Wb_pt_order < num_jet_to_select,
              Wc_pt_order < num_jet_to_select,)



#%% Save the features in a numpy matrix

df=pd.DataFrame()
#Save only the events in which the leptonic b jet is distant less than 0.4 from the LHE leptonic b jet
events.Jet=events.Jet[:,:num_jet_to_select]
Jet_pt=events.Jet.pt[mask]
Jet_eta=events.Jet.eta[mask]
Jet_phi=events.Jet.phi[mask]
Jet_mass = events.Jet.mass[mask]
Jet_btag=events.Jet.btagDeepFlavB[mask]
Jet_CvBtag=events.Jet.btagDeepFlavCvB[mask]
Jet_CvLtag=events.Jet.btagDeepFlavCvL[mask]
dPhi_Jet_mu=events.Jet.delta_phi(events.Muon[:,0])[mask]
dPhi_Jet_nu=events.Jet.delta_phi(events.MET)[mask]
dEta_Jet_mu=(events.Jet.eta-events.Muon[:,0].eta)[mask]
dEta_Jet_nu=(events.Jet.eta-nu_4Vect.eta)[mask]

T_mass=(events.Jet+events.Muon[:,0]+nu_4Vect)[mask].mass

#min/max dphi deta jets
min_dPhi_Jets=minmax_delta_jets(events.Jet[mask], "min",metric="phi",level="jet",flatten_axis=2,numpy=False)
max_dPhi_Jets=minmax_delta_jets(events.Jet[mask], "max",metric="phi",level="jet",flatten_axis=2,numpy=False)

min_dEta_Jets=minmax_delta_jets(events.Jet[mask], "min",metric="eta",level="jet",flatten_axis=2,numpy=False)
max_dEta_Jets=minmax_delta_jets(events.Jet[mask], "max",metric="eta",level="jet",flatten_axis=2,numpy=False)


#Event id
#nJets_per_event = ak.count(events.Jet.pt, axis=-1)
#event_id=np.repeat(np.arange(len(events.Jet.pt))[mask], nJets_per_event[mask])

event_id=ak.Array(np.arange(len(events.Jet.pt))[mask])



#LABEL: 1=Jet from Leptonic T, 0=Others

#[1]: return the metric (min delta_r)
#[:,[2,5]][leptonic_LHE_mask]: Select the leptonic b jet from the LHE
#[mask]: Select only the events in which the leptonic b jet is distant less than 0.4 from the LHE jet
deltaR_jet_LHE=events.LHEPart.nearest(events.Jet,return_metric=True)[1][:, [2, 5]][leptonic_LHE_mask][mask]

label = (events.Jet.delta_r(bLept_LHE)[
         mask] == deltaR_jet_LHE)
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
perm2=np.random.permutation(num_jet_to_select)
i, j = features_df.index.levels
idx = pd.IndexSlice

features_df = features_df.loc[i[perm1]]
features_df = features_df.loc[idx[i, j[perm2]],:]
features_df=features_df.dropna()


#features_df.to_pickle("Jet_features.pkl",compression="bz2")

# %%
