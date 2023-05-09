#%%

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from utils import build
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from JPAmodel.dataset import DatasetBuilder
import torch


n_jet=7

#signal_path="../TTbarSemileptonic_cbOnly_pruned_optimized.root"
signal_path="../../../root_files/signal_background/BigMuons_MuonCuts.root"
signal=NanoEventsFactory.from_root(
    signal_path,
    schemaclass=NanoAODSchema
).events()


#background_path="../TTbarSemileptonic_Nocb_pruned_optimized.root"
background_path="../../../root_files/signal_background/BigMuons_jj_MuonCuts.root"
background=NanoEventsFactory.from_root(
    background_path,
    schemaclass=NanoAODSchema
).events()


powheg_path="../../../root_files/signal_background/TTbarSemilept_MuonCuts_powheg.root"
powheg=NanoEventsFactory.from_root(
    powheg_path,
    schemaclass=NanoAODSchema
).events()


#%%
background_matrix,col_labels=build(background,0,num_jet_to_select=n_jet,LHELept=13)
signal_matrix,_=build(signal,1,num_jet_to_select=n_jet,LHELept=13)

n_powheg_train=5500000
powheg_train_matrix,_=build(powheg[:n_powheg_train],0,num_jet_to_select=n_jet,LHELept=13)
powheg_matrix,_=build(powheg[n_powheg_train:],0,num_jet_to_select=n_jet)

event_matrix=np.vstack([signal_matrix,background_matrix,powheg_train_matrix])

train_df,test_df = train_test_split(pd.DataFrame(event_matrix, columns=col_labels),test_size=0.15,shuffle=True)
powheg_df = pd.DataFrame(powheg_matrix, columns=col_labels).sample(frac=1)

data_builder=DatasetBuilder(train_df,n_jet)
train_dataset=data_builder.train_dataset
test_dataset=data_builder.build_dataset(test_df,n_jet)
powheg_dataset=data_builder.build_dataset(powheg_df,n_jet)

torch.save(train_dataset,"../../../root_files/signal_background/train_dataset.pt")
torch.save(test_dataset,"../../../root_files/signal_background/test_dataset.pt")
torch.save(powheg_dataset,"../../../root_files/signal_background/powheg_dataset.pt")

# %%
