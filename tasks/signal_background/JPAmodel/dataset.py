import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def build_datasets(df,jets_per_event,test_size=0.15,normalize=True):
            
        mu_data=df.filter(regex="Muon.*(pt|eta|phi)").to_numpy()
        nu_data=df.filter(regex="MET.*(pt|eta|phi)").to_numpy()
        jet_data=df.filter(regex="Jet.*(pt|eta|phi|btagDeepFlavCvB|btagDeepFlavCvL|Tmass)").to_numpy()
        label=df.filter(regex="(label).*").astype(int).to_numpy()



        mu_data=np.reshape(mu_data, (mu_data.shape[0],1, mu_data.shape[1]))
        nu_data=np.reshape(nu_data, (nu_data.shape[0],1, nu_data.shape[1]))
        jet_data=np.reshape(jet_data,
                            (jet_data.shape[0],
                            jets_per_event,
                            jet_data.shape[1]//jets_per_event))

        mu_data = torch.tensor(mu_data, dtype=torch.float32)
        nu_data = torch.tensor(nu_data, dtype=torch.float32)
        jet_data = torch.tensor(jet_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        if test_size is not None:
            mu_train, mu_test, nu_train, nu_test,\
            jet_train,jet_test,y_train,y_test=train_test_split(mu_data,nu_data,jet_data,label,test_size=test_size,shuffle=True)
            return EventsDataset(mu_train,nu_train,jet_train,y_train,normalize=normalize),EventsDataset(mu_test,nu_test,jet_test,y_test,normalize=normalize)
        else:
            return EventsDataset(mu_data,nu_data,jet_data,label,normalize=normalize)
        
class EventsDataset(Dataset):
    def __init__(self, mu_data, nu_data, jet_data, label,normalize=True):
        if normalize:
            temp_jet_df=pd.DataFrame(torch.flatten(jet_data,end_dim=1).numpy())
            jet_mean=torch.tensor(temp_jet_df[temp_jet_df!=0].mean().to_numpy(),dtype=torch.float32)
            jet_std=torch.tensor(temp_jet_df[temp_jet_df!=0].std().to_numpy(),dtype=torch.float32)

            del temp_jet_df

                    
            
            
            self.mu_data = (mu_data-torch.mean(mu_data))/(torch.std(mu_data)+1e-5)
            self.nu_data = (nu_data-torch.mean(nu_data))/(torch.std(nu_data)+1e-5)
            self.jet_data = (jet_data-jet_mean)/(jet_std+1e-5)
        else:
            self.mu_data = mu_data
            self.nu_data = nu_data
            self.jet_data = jet_data
        self.label = label

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
        
        
def loader(dataset,batch_size):
    n_batch=int(np.ceil(len(dataset)/batch_size))
    for i in range(n_batch):
        yield dataset[i*batch_size:(i+1)*batch_size]

