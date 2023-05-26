#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from utils import build
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from JPAmodel.dataset import EventsDataset, DataBuilder
import argparse
import torch

parser = argparse.ArgumentParser(description='Save pytorch dataset (ONLY MADGRAPH)')
parser.add_argument('-i', '--input', help='Root file input')
parser.add_argument('-o', '--output', help='Save path')
args = parser.parse_args()
data_path=args.input
save_path=args.output

n_jet=7



powheg_path="../../../root_files/signal_background/Muon/powheg_Muon_dataset.pt"
powheg=torch.load(powheg_path)
stats_dict=powheg.stats_dict
del powheg



data=NanoEventsFactory.from_root(
    data_path,
    schemaclass=NanoAODSchema
).events()

data_matrix,col_labels=build(data,0,num_jet_to_select=n_jet)
data_df = pd.DataFrame(data_matrix, columns=col_labels).sample(frac=1)

builder=DataBuilder(stats_dict)
dataset=builder.build_dataset(data_df,n_jet,LHE_pdgId_madgraph=data.LHEPart.pdgId)
torch.save(dataset,save_path)