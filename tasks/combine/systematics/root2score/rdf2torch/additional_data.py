import torch
import awkward as ak
import numpy as np
from root2score.rdf2torch.torch_dataset import EventsDataset

def LeptLabel(pdgId):

    lept_label=torch.ones(ak.num(pdgId,axis=0),1)
    
    MuonMask=ak.sum((pdgId)==13,axis=1).to_numpy()
    ElectronMask=ak.sum((pdgId)==11,axis=1).to_numpy()
    TauMask=ak.sum((pdgId)==15,axis=1).to_numpy()
    AMuonMask=ak.sum((pdgId)==-13,axis=1).to_numpy()
    AElectronMask=ak.sum((pdgId)==-11,axis=1).to_numpy()
    ATauMask=ak.sum((pdgId)==-15,axis=1).to_numpy()
    
    lept_label[MuonMask==1]*=13
    lept_label[AMuonMask==1]*=-13
    lept_label[ElectronMask==1]*=11
    lept_label[AElectronMask==1]*=-11
    lept_label[TauMask==1]*=15
    lept_label[ATauMask==1]*=-15
    
    return lept_label

def HadDecay(pdgId,generator):
    assert generator in ["powheg","madgraph"]

    n=len(pdgId)
    if generator=="powheg":
        try:
            had_decay=pdgId[:,4:]
            had_decay=had_decay[np.abs(had_decay)<5]
            had_decay=torch.tensor(had_decay.to_numpy(),dtype=int)
        except ValueError:
            had_decay=torch.tensor(np.zeros((n)))
    elif generator=="madgraph":
        had_decay=pdgId[:,[3,4,6,7]]
        had_decay=had_decay[np.abs(had_decay)<6]
        had_decay=torch.tensor(had_decay.to_numpy(),dtype=int)
        had_decay=had_decay.reshape(n,2)
    
    #print(had_decay.shape)
    
    return had_decay
    
def AdditionalPartons(pdgId,generator):
    assert generator in ["powheg","madgraph"]
    n_partons=ak.num(pdgId,axis=1).to_numpy()
    additional_partons=np.zeros(len(pdgId))
    if generator=="powheg":
        additional_partons[n_partons==9]=pdgId[n_partons==9,2].to_numpy()
    elif generator=="madgraph":
        additional_partons=ak.pad_none(pdgId,11,clip=True,axis=1)
        additional_partons=ak.fill_none(additional_partons,0).to_numpy()[:,8:]
    return torch.tensor(additional_partons,dtype=int)




def add_additional_data(dataset, pdgId, additional_list,generator=None):
    assert generator in ["powheg","madgraph"]
    for info in additional_list:
        assert info in ["LeptLabel","HadDecay","AdditionalPartons","TopLept"]
        if info=="LeptLabel":
            information="Product of pdgId of leptons (1=no leptons)"
            dataset.add_data(info,LeptLabel(pdgId),[information])
        elif info=="HadDecay":
            information="Couple of quarks pdgid from top hadronic decay"
            dataset.add_data(info,HadDecay(pdgId,generator=generator),[information])
        elif info=="AdditionalPartons":
            information="PdgIds of additional partons"
            dataset.add_data(info,AdditionalPartons(pdgId,generator=generator),[information])

    return dataset
            