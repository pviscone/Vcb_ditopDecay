from torch.utils.data import Dataset
import torch
import numpy as np
import copy
class EventsDataset(Dataset):
    """Pytorch dataset based on dictionaries

    Args:
        Dataset (_type_): pytorch dataset
    """
    def __init__(self):
        """ data: dictionary of tensors
            info: dictionary of column info
            additional_info: dictionary of additional info
        """
        self.data={}
        self.info={}
        self.additional_info={}

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        key=list(self.data.keys())[0]
        return len(self.data[key])

    
    def __getitem__(self, *key_idx):
        """ Return the element of the data dictionar
            dataset[key,idx/slice]
        Returns:
            torch.tensor: tensor
        """
        return self.data[key_idx[0][0]][key_idx[0][1]]
    
    def add_data(self,name,data,column_info):
        """Add data to the data dictionary

        Args:
            name (string):key of the data dictionary
            data (torch.tensor): tensor data
            column_info (List[String]): description of the data columns
        """
        self.data[name]=data
        self.info[name]=column_info
    
    def add_additional_info(self,name,data):
        """ Add auxiliary informations

        Args:
            name (string): key of the dictionary
            data (obj): value of the dictionary
        """
        self.additional_info[name]=data
    
    def to(self,device):
        """Move the data to the device

        Args:
            device (_type_): pytorch device
        """
        for key in self.data.keys():
            if self.data[key] is None:
                continue
            self.data[key]=self.data[key].to(device)
    
    def mask(self,mask,retrieve=False):
        """Mask the data

        Args:
            mask (Tensor[Bool]): tensor of booleans of the same length of the dataset
        """
        mask=mask.squeeze()
        if retrieve is True:
            dataset=copy.deepcopy(self)
            dataset.mask(mask)
            return dataset
        else:
            for key in self.data.keys():
                if self.data[key] is None:
                    continue
                self.data[key]=self.data[key][mask].clone()
            
    def get(self,key):
        """Get the data calling the key

        Args:
            key (string): Name of the column contained in dataset.info[object]

        Returns:
            torch.tensor: Column of the dataset
        """
        obj=key.split("_")[0]
        idx=np.where(np.array(self.info[obj])==key)[0]
        return self.data[obj][:,:,idx].squeeze()
    
    def shuffle(self,retrieve=False):
        """shuffle the data
        """
        if retrieve is True:
            dataset=copy.deepcopy(self)
            dataset.shuffle()
            return dataset
        else:
            keys= list(self.data.keys())
            idx=torch.randperm(self.data[keys[0]].shape[0])
            for key in keys:
                if self.data[key] is None:
                    continue
                self.data[key]=self.data[key][idx].clone()
            
    def get_batch(self,start,end):
        """Return a batch of the dataset

        Args:
            start (int): start index
            end (int): end index

        Returns:
            EventsDataset: dataset containing the batch
        """
        batch=EventsDataset()
        batch.info=copy.copy(self.info)
        batch.additional_info=copy.copy(self.additional_info)
        for key in self.data.keys():
            if self.data[key] is None:
                continue
            batch.data[key]=self.data[key][start:end].clone()
        return batch
    
    def slice(self,start,end):
        """Select a slice of the data (on the first index) for all the keys
        (It overrites the existing data)
        

        Args:
            start (int): start index
            end (int): final index
        """

        for key in self.data.keys():
            if self.data[key] is None:
                continue
            self.data[key]=self.data[key][start:end].clone()
    
    def cat(self,dataset, retrieve=False):
        
        if retrieve is True:
            ds=copy.deepcopy(self)
            ds.cat(dataset)
            return ds
        else:
            for key in self.data.keys():
                if (self.data[key] is None or dataset.data[key] is None):
                    continue
                if (key=="AdditionalPartons" or key=="HadDecay"):
                    if (dataset.data[key].ndim==1):
                        dataset.data[key]=(dataset.data[key]).unsqueeze(dim=1)
                    if (self.data[key].ndim==1):
                        self.data[key]=(self.data[key]).unsqueeze(dim=1)
                    d1=dataset.data[key].shape[1]
                    d2=self.data[key].shape[1]
                    if d1<d2:
                        dataset.data[key]=torch.nn.functional.pad(dataset.data[key],(0,d2-d1))
                    else:
                        self.data[key]=torch.nn.functional.pad(self.data[key],(0,d1-d2))
                    
                    
                self.data[key]=torch.cat((self.data[key],dataset.data[key]),dim=0)
            
    def ls(self):
        """Print the keys and the shape of the data dictionary
        """
        for key in self.data.keys():
            if self.data[key] is None:
                continue
            print(f"key: {key}, shape: {self.data[key].shape}")
        