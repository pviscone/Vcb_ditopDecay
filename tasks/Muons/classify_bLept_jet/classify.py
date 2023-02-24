#%% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import MLP_model
import importlib
importlib.reload(MLP_model)
MLP=MLP_model.MLP

#%% Load data and create variables

events = NanoEventsFactory.from_root(
    "../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

