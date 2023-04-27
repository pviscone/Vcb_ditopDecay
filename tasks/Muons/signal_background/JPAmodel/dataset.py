import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EventsDataset(Dataset):
    def __init__(self, mu_data, nu_data, jet_data, label):
        temp_jet_df=pd.DataFrame(torch.flatten(jet_data,end_dim=1).numpy())
        jet_mean=torch.tensor(temp_jet_df[temp_jet_df!=0].mean().to_numpy(),dtype=torch.float32)
        jet_std=torch.tensor(temp_jet_df[temp_jet_df!=0].std().to_numpy(),dtype=torch.float32)

        del temp_jet_df

                
        
        
        self.mu_data = (mu_data-torch.mean(mu_data))/(torch.std(mu_data)+1e-5)
        self.nu_data = (nu_data-torch.mean(nu_data))/(torch.std(nu_data)+1e-5)
        self.jet_data = (jet_data-jet_mean)/(jet_std+1e-5)
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

