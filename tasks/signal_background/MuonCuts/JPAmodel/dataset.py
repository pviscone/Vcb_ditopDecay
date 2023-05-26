import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import awkward as ak

class DataBuilder():
    def __init__(self,stats):
        self.mu_mean=stats["mu_mean"]
        self.nu_mean=stats["nu_mean"]
        self.jet_mean=stats["jet_mean"]
        self.mu_std=stats["mu_std"]
        self.nu_std=stats["nu_std"]
        self.jet_std=stats["jet_std"]

    def build_dataset(self,df,jets_per_event,LHE_pdgId_powheg=None,LHE_pdgId_madgraph=None):
        mu_data=df.filter(regex="Muon.*(pt|eta|phi)").to_numpy()
        nu_data=df.filter(regex="MET.*(pt|eta|phi)").to_numpy()
        jet_data=df.filter(regex="Jet.*(pt|eta|phi|btagDeepFlavCvB|btagDeepFlavCvL|TLeptMass|THadMass|WHadMass)").to_numpy()
        label=df["label"].astype(int).to_numpy()
        Lept_label=df.filter(regex="(Lept_label).*").astype(int).to_numpy()

        mu_data=np.reshape(mu_data, (mu_data.shape[0],1, mu_data.shape[1]))
        nu_data=np.reshape(nu_data, (nu_data.shape[0],1, nu_data.shape[1]))
        jet_data=np.reshape(jet_data,
                            (jet_data.shape[0],
                            jets_per_event,
                            jet_data.shape[1]//jets_per_event))

        mu_data = torch.tensor(mu_data, dtype=torch.float32)
        nu_data = torch.tensor(nu_data, dtype=torch.float32)
        jet_data = torch.tensor(jet_data, dtype=torch.float32)
        missing_mask=jet_data==-10
        label = torch.tensor(label, dtype=torch.long)
        
    
        mu_data=torch.tanh(0.01*((mu_data-self.mu_mean)/(self.mu_std+1e-6)))
        nu_data=torch.tanh(0.01*((nu_data-self.nu_mean)/(self.nu_std+1e-6)))
        jet_data=torch.tanh(0.01*((jet_data-self.jet_mean)/(self.jet_std+1e-6)))
        jet_data[missing_mask]=-1.01
        stats_dict={"mu_mean":self.mu_mean,"mu_std":self.mu_std,"nu_mean":self.nu_mean,"nu_std":self.nu_std,"jet_mean":self.jet_mean,"jet_std":self.jet_std}

        return EventsDataset(mu_data,nu_data,jet_data,label,Lept_label,stats_dict,LHE_pdgId_powheg,LHE_pdgId_madgraph)





class DatasetBuilder():
    def __init__(self,df,jets_per_event):
        train_dataset,stats=self.build_dataset(df,jets_per_event,stats=True)
        self.mu_mean=stats["mu_mean"]
        self.nu_mean=stats["nu_mean"]
        self.jet_mean=stats["jet_mean"]
        self.mu_std=stats["mu_std"]
        self.nu_std=stats["nu_std"]
        self.jet_std=stats["jet_std"]
        self.train_dataset=train_dataset
        
        

    def build_dataset(self,df,jets_per_event,stats=False,LHE_pdgId_powheg=None):
            mu_data=df.filter(regex="Muon.*(pt|eta|phi)").to_numpy()
            nu_data=df.filter(regex="MET.*(pt|eta|phi)").to_numpy()
            jet_data=df.filter(regex="Jet.*(pt|eta|phi|btagDeepFlavCvB|btagDeepFlavCvL|TLeptMass|THadMass|WHadMass)").to_numpy()
            label=df["label"].astype(int).to_numpy()
            Lept_label=df.filter(regex="(Lept_label).*").astype(int).to_numpy()

            mu_data=np.reshape(mu_data, (mu_data.shape[0],1, mu_data.shape[1]))
            nu_data=np.reshape(nu_data, (nu_data.shape[0],1, nu_data.shape[1]))
            jet_data=np.reshape(jet_data,
                                (jet_data.shape[0],
                                jets_per_event,
                                jet_data.shape[1]//jets_per_event))

            mu_data = torch.tensor(mu_data, dtype=torch.float32)
            nu_data = torch.tensor(nu_data, dtype=torch.float32)
            jet_data = torch.tensor(jet_data, dtype=torch.float32)
            missing_mask=jet_data==-10
            label = torch.tensor(label, dtype=torch.long)
            
            if stats is True:
                temp_jet_df=pd.DataFrame(torch.flatten(jet_data,end_dim=1).numpy())
                jet_mean=torch.tensor(temp_jet_df[temp_jet_df!=-10].mean().to_numpy(),dtype=torch.float32)
                jet_std=torch.tensor(temp_jet_df[temp_jet_df!=-10].std().to_numpy(),dtype=torch.float32)

                del temp_jet_df
                mu_mean=torch.mean(mu_data,axis=0)
                mu_std=torch.std(mu_data,axis=0)
                nu_mean=torch.mean(nu_data,axis=0)
                nu_std=torch.std(nu_data,axis=0)
                mu_data=torch.tanh(0.01*((mu_data-mu_mean)/(mu_std+1e-6)))
                nu_data=torch.tanh(0.01*((nu_data-nu_mean)/(nu_std+1e-6)))
                jet_data=torch.tanh(0.01*((jet_data-jet_mean)/(jet_std+1e-6)))
                jet_data[missing_mask]=-1.01
                stats_dict={"mu_mean":mu_mean,"mu_std":mu_std,"nu_mean":nu_mean,"nu_std":nu_std,"jet_mean":jet_mean,"jet_std":jet_std}
                return EventsDataset(mu_data,nu_data,jet_data,label,Lept_label,stats_dict),stats_dict
            else:
                mu_data=torch.tanh(0.01*((mu_data-self.mu_mean)/(self.mu_std+1e-6)))
                nu_data=torch.tanh(0.01*((nu_data-self.nu_mean)/(self.nu_std+1e-6)))
                jet_data=torch.tanh(0.01*((jet_data-self.jet_mean)/(self.jet_std+1e-6)))
                jet_data[missing_mask]=-1.01
                stats_dict={"mu_mean":self.mu_mean,"mu_std":self.mu_std,"nu_mean":self.nu_mean,"nu_std":self.nu_std,"jet_mean":self.jet_mean,"jet_std":self.jet_std}
                if LHE_pdgId_powheg is not None:
                    return EventsDataset(mu_data,nu_data,jet_data,label,Lept_label,stats_dict,LHE_pdgId_powheg)
                else:
                    return EventsDataset(mu_data,nu_data,jet_data,label,Lept_label,stats_dict)
            
        
class EventsDataset(Dataset):
    def __init__(self, mu_data, nu_data, jet_data, label,Lept_label,stats_dict,LHE_pdgId_powheg=None,LHE_pdgId_madgraph=None):
        self.mu_data = mu_data
        self.nu_data = nu_data
        self.jet_data = jet_data
        self.label = label
        self.Lept_label=Lept_label
        self.stats_dict=stats_dict
        if LHE_pdgId_powheg is not None:
            additional_parton=np.zeros(len(LHE_pdgId_powheg))
            n_partons=ak.num(LHE_pdgId_powheg,axis=1).to_numpy()
            additional_parton[n_partons==9]=LHE_pdgId_powheg[n_partons==9,2].to_numpy()
            self.additional_parton=torch.tensor(additional_parton,dtype=int)
            had_decay=LHE_pdgId_powheg[:,4:]
            had_decay=had_decay[np.abs(had_decay)<5]
            self.had_decay=torch.tensor(had_decay.to_numpy(),dtype=int)
        if LHE_pdgId_madgraph is not None:
            lhe=ak.pad_none(LHE_pdgId_madgraph,11,clip=True,axis=1)
            lhe=ak.fill_none(lhe,0).to_numpy()
            self.additional_parton=torch.tensor(lhe[:,8:])
            self.had_decay=torch.tensor(lhe[:,3:8][np.abs(lhe[:,3:8])<5],dtype=int)
            

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        mu = self.mu_data[idx]
        nu = self.nu_data[idx]
        jet = self.jet_data[idx]
        label = self.label[idx]
        return mu, nu, jet, label
    
    def to(self,device):
        self.mu_data=self.mu_data.to(device)
        self.nu_data=self.nu_data.to(device)
        self.jet_data=self.jet_data.to(device)
        self.label=self.label.to(device)
    
    def slice(self,start,end):
        self.mu_data=self.mu_data[start:end]
        self.nu_data=self.nu_data[start:end]
        self.jet_data=self.jet_data[start:end]
        self.label=self.label[start:end]
        self.Lept_label=self.Lept_label[start:end]
        
        
def loader(dataset,batch_size):
    n_batch=int(np.ceil(len(dataset)/batch_size))
    for i in range(n_batch):
        yield dataset[i*batch_size:(i+1)*batch_size]

