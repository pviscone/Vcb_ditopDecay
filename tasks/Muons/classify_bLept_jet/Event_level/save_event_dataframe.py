#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    "../../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

def padded_matrix(ak_array,pad):
    masked_array = ak.pad_none(ak_array, pad, clip=True).to_numpy()
    masked_array.data[masked_array.mask] = 0
    return masked_array.data


def column(ak_array):
    return np.atleast_2d(ak_array.to_numpy(allow_missing=False)).T

def build_matrix(obj,variable_list):
    column_list=[]
    col_labels=[]
    for var in variable_list:
        col_labels.append(f"{obj}_{var}")
        exec(f"column_list.append(column(events.{obj}.{var}))")
    return np.hstack(column_list), col_labels


def alternate_column(matrix_list):
    num_particles = matrix_list[0].shape[1]
    num_features = len(matrix_list)
    final_matrix = np.empty(
        (matrix_list[0].shape[0], num_particles*num_features))
    for feature in range(num_features):
        for obj in range(num_particles):
            final_matrix[:, num_features*obj +
                         feature] = matrix_list[feature][:, obj]
    return final_matrix

def pad_and_alternate(obj,variable_list,pad):
    matrix_list=[]
    col_labels=[]
    for var in variable_list:
        if var=="Tmass":
            matrix_list.append(padded_matrix(events.Tmass,pad))
        else:
            exec(f"matrix_list.append(padded_matrix(events.{obj}.{var},{pad}))")
    for i in range(pad):
        for var in variable_list:
            col_labels.append(f"{obj}{i}_{var}")
    return alternate_column(matrix_list), col_labels

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

#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)

bLept_LHE=events.LHEPart[:,[2,5]][leptonic_LHE_mask]
bLept_Jet,deltaR = events.LHEPart.nearest(events.Jet,return_metric=True)
bLept_Jet = bLept_Jet[:, [2, 5]][leptonic_LHE_mask]
deltaR=deltaR[:, [2, 5]][leptonic_LHE_mask]



plt.title("Order in pt of the bLept jet")
bLept_pt_order = ak.argmax(events.Jet.pt == bLept_Jet.pt, axis=1)
plt.hist(bLept_pt_order)
plt.yscale("log")
plt.xlabel("Order in pt")


num_jet_to_select=7
efficiency = len(bLept_pt_order[np.bitwise_and(bLept_pt_order <
                 num_jet_to_select,deltaR<0.4)])/len(bLept_pt_order[deltaR<0.4])
print(f"Percentage of events with the bLept jet in the first {num_jet_to_select} jets:{efficiency}")

mask= np.bitwise_and(deltaR<0.4, bLept_pt_order < num_jet_to_select)

#%%
events.Jet=events.Jet[:, :num_jet_to_select]
events = events[mask]
events.Neutrino = nu_4Vect[mask]
bLept_LHE = bLept_LHE[mask]
bLept_Jet = bLept_Jet[mask]


events.Muon = events.Muon[:, 0]
events.Neutrino.Wmass = (events.Muon+events.Neutrino).mass
events.Tmass = (events.Neutrino+events.Muon+events.Jet).mass


#%%
col_labels=[]

muon_matrix,mu_labels=build_matrix("Muon",["pt","eta","phi","mass"])
nu_matrix,nu_labels=build_matrix("Neutrino",["pt","eta","phi","mass","Wmass"])
jet_matrix,jet_labels=pad_and_alternate("Jet",["pt","eta","phi","mass","btagDeepFlavB","btagDeepFlavCvB","btagDeepFlavCvL","area","Tmass"],num_jet_to_select)

label = column(bLept_pt_order[mask])
col_labels=mu_labels+nu_labels+jet_labels+["label"]

event_matrix=np.hstack([muon_matrix,nu_matrix,jet_matrix,label])

event_df=pd.DataFrame(event_matrix,columns=col_labels)
event_df=event_df.sample(frac=1)
#event_df.to_pickle("./event_df.pkl", compression="bz2")
# %%
