#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from utils import pad_and_alternate, build_matrix,select_muon_events
import numpy as np
import pandas as pd

#%%
signal=NanoEventsFactory.from_root(
    "../TTbarSemileptonic_cbOnly_pruned_optimized.root",
    schemaclass=NanoAODSchema
).events()

background=NanoEventsFactory.from_root(
    "../TTbarSemileptonic_Nocb.root",
    schemaclass=NanoAODSchema
).events()

num_jet_to_select=7

signal=select_muon_events(signal,num_jet_to_select)
background=select_muon_events(background,num_jet_to_select)

#%%

Mu_feature=["pt", "eta", "phi"]
Nu_feature=["pt", "eta", "phi"]
Jet_feature=["pt", "eta", "phi", "btagDeepFlavB",
             "btagDeepFlavCvB", "btagDeepFlavCvL", "Tmass"]

signal_muon_matrix, mu_labels = build_matrix(signal,"Muon", Mu_feature)
signal_nu_matrix, nu_labels = build_matrix(signal,"MET", Nu_feature)
signal_jet_matrix, jet_labels = pad_and_alternate(signal,"Jet",
                                                    Jet_feature,
                                                    num_jet_to_select)
background_muon_matrix, mu_labels = build_matrix(background,"Muon", Mu_feature)
background_nu_matrix, nu_labels = build_matrix(background,"MET", Nu_feature)
background_jet_matrix, jet_labels = pad_and_alternate(background,"Jet",
                                                    Jet_feature,
                                                    num_jet_to_select)

signal_matrix=np.hstack([
    signal_muon_matrix,
    signal_nu_matrix,
    signal_jet_matrix,
    np.ones((signal_muon_matrix.shape[0],1)),])

background_matrix=np.hstack([
    background_muon_matrix,
    background_nu_matrix,
    background_jet_matrix,
    np.zeros((background_muon_matrix.shape[0],1))])

event_matrix=np.vstack([signal_matrix,background_matrix])
col_labels=mu_labels+nu_labels+jet_labels+["label"]
event_df = pd.DataFrame(event_matrix, columns=col_labels).sample(frac=1)
event_df.to_pickle("event_df.pkl")