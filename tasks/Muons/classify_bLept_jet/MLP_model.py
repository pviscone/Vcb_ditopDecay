import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
#from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

#enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
  
cpu=torch.device("cpu")
device = torch.device(dev)



def custom_collate_fn(batch):
    batch = batch.to(device, non_blocking=True)
    return batch


class MLP(torch.nn.Module):

    def __init__(self,hidden_arch=[10,10],batch_size=1,
                 shuffle=False,
                 x_train=None,y_train=None,x_test=None,y_test=None,optim={}):
        super().__init__()

        self.loss_fn=torch.nn.BCELoss(weight=torch.tensor([6,1],device=device))

        
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.hidden_arch=hidden_arch

        self.mean=x_train.mean(axis=0)
        self.std=x_train.std(axis=0)


        self.shuffle_at_each_epoch=shuffle
        self.batch_size=batch_size
        self.n_batch=np.ceil(len(x_train)/batch_size).astype(int)
        self.test_loss=[]

        self.train_loss=[]

        self.n_inputs=x_train.shape[1]
        self.n_outputs=y_train.shape[1]
        
        
        #self.layers = torch.jit.trace(torch.nn.ModuleList())
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(self.n_inputs, hidden_arch[0]))
        
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers.append(torch.nn.ReLU())
        for in_neurons,out_neurons in zip(self.hidden_arch[:-1],self.hidden_arch[1:]):
            self.layers.append(torch.nn.Linear(in_neurons,out_neurons))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(torch.nn.ReLU())
        
        self.layers.append(torch.nn.Linear(self.hidden_arch[-1],self.n_outputs))
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        if self.n_outputs==1:
            self.layers.append(torch.nn.Sigmoid())
        else:
            self.layers.append(torch.nn.Softmax(dim=1))
        
    
        self.optim_dict=optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)
        
        self.false_negative=[]
        self.false_positive=[]

    def forward(self,x):
        x=(x-self.mean)/self.std
        for layer in self.layers:
            x=layer(x)
        return x
    
    def error(self,type=None,dataset=None):
        if type not in ["I","II"]:
            raise ValueError("type must be either train or test")
        if dataset not in ["train","test"]:
            raise ValueError("dataset must be either train or test")
        
        if type=="I":
            true=torch.tensor([1,0],device=device)
            predicted=torch.tensor([0,1],device=device)
        elif type=="II":
            true=torch.tensor([0,1],device=device)
            predicted=torch.tensor([1,0],device=device)
        
        if dataset=="train":
            x=self.x_train
            y=(self.y_train)
        elif dataset=="test":
            x=self.x_test
            y=(self.y_test)
        
        mask = (y == true)[:,0]
        y_pred=((self(x[mask]).round())==predicted)[:,0]
        return y_pred.sum()/len(y_pred)
        
    
    def train_loop(self,epochs):
        epoch_loop=tqdm(range(epochs),desc="epoch")
        batch_loop=tqdm(range(self.n_batch),desc="batch")
        for epoch in epoch_loop:
            self.train()
            x_train=self.x_train
            y_train=self.y_train
            if self.shuffle_at_each_epoch:
                perm=torch.randperm(self.x_train.size()[0])
                x_train=x_train[perm]
                y_train=y_train[perm]
            
            #for x_batch, y_batch in self.data_loader:
            for i in batch_loop:
                if((i+1)*self.batch_size<len(x_train)):
                    x_batch=x_train[i*self.batch_size:(i+1)*self.batch_size]
                    y_batch = y_train[i *self.batch_size:(i+1)*self.batch_size]
                else:
                    x_batch=x_train[i*self.batch_size:len(x_train)]
                    y_batch = y_train[i*self.batch_size:len(y_train)]
                y_logits=self.forward(x_batch).squeeze()
                train_loss_step=self.loss_fn(y_logits,y_batch)
                self.optimizer.zero_grad()
                train_loss_step.backward()
                self.optimizer.step()
            
            self.eval()
            with torch.inference_mode():
                
                test_logits=self.forward(self.x_test).squeeze()
                test_pred=test_logits.round()
                
                test_loss_step=self.loss_fn(test_logits,self.y_test.squeeze())

                
                self.test_loss.append(test_loss_step.to(cpu).numpy())

                self.train_loss.append(train_loss_step.to(cpu).numpy())

                
                self.false_positive.append(self.error(type="I",dataset="test").to(cpu).numpy())
                
                self.false_negative.append(self.error(type="II", dataset="test").to(cpu).numpy())

                    
    def loss_plot(self):
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.plot(self.train_loss,label="train")
        plt.plot(self.test_loss,label="test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("BCE loss")
        plt.subplot(132)
        plt.plot(self.false_negative,label="false negative")
        plt.plot(self.false_positive,label="false positive")
        plt.title("Type I and II error: TEST dataset")
        plt.legend()
        plt.xlabel("Epochs")
        plt.subplot(133)
        with torch.no_grad():
            confusion_matrix_test = torch.tensor(
                [[1-self.false_positive[-1], self.false_positive[-1]], [self.false_negative[-1], 1-self.false_negative[-1]]])

            confusion_matrix_test = pd.DataFrame(confusion_matrix_test, index=[
                                                "true negative", "true positive"], columns=["predicted negative", "predicted positive"])

            sn.heatmap(confusion_matrix_test, annot=True, fmt="g", cmap="viridis")
            

    def evaluate_on_events(self):
        pred = self(self.x_test)[:, 1]

        res = pd.DataFrame({"pred": pred.detach().cpu().numpy(),
                            "label": self.y_test[:, 1].cpu().numpy()})

        selected_jets_idx = res.groupby(level=0)["pred"].idxmax()

        efficiency = res["label"][selected_jets_idx].sum()/len(selected_jets_idx)
        print(f"Efficiency: {efficiency*100:.2f}%")
        return efficiency
