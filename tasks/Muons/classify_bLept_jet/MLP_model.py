import torch
import numpy as np
import wandb
from typing import List

#enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
  
cpu=torch.device("cpu")
device = torch.device(dev)


class MLP(torch.nn.Module):

    def __init__(self,hidden_arch=[10,10],batch_size=1,x_train=None,y_train=None,x_test=None,y_test=None,optim={}):
        super().__init__()

        self.loss_fn=torch.nn.BCELoss(weight=torch.tensor([5,1],device=device))

        
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.hidden_arch=hidden_arch

        tensor_dataset=torch.utils.data.TensorDataset(self.x_train, self.y_train)

        self.data_loader=torch.utils.data.DataLoader(tensor_dataset,batch_size=batch_size)

        
        
        
        
        self.test_loss=[]

        self.train_loss=[]

        self.n_inputs=x_train.shape[1]
        self.n_outputs=y_train.shape[1]
        
        
        #self.layers = torch.jit.trace(torch.nn.ModuleList())
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(self.n_inputs, hidden_arch[0]))
        
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        
        self.layers.append(torch.nn.Sigmoid())
        for in_neurons,out_neurons in zip(self.hidden_arch[:-1],self.hidden_arch[1:]):
            self.layers.append(torch.nn.Linear(in_neurons,out_neurons))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(torch.nn.Sigmoid())
        
        self.layers.append(torch.nn.Linear(self.hidden_arch[-1],self.n_outputs))
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        if self.n_outputs==1:
            self.layers.append(torch.nn.Sigmoid())
        else:
            self.layers.append(torch.nn.Softmax(dim=1))
        
    
        self.optim_dict=optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)
        
        self.wandb=False

        
        self.false_negative=[]
        self.false_positive=[]
        

    def wandb_init(self,project,config,**log_kwargs):
        self.wandb=True
        wandb.init(project=project, config=config)
        self.wandb_log_kwargs=log_kwargs

    def forward(self,x):
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
        
        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in self.data_loader:
                
                
                
                y_logits=self.forward(x_batch).squeeze()
                y_pred=y_logits.round()
                
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
                
                if self.wandb:
                    wandb.log(
                        {"test_loss": test_loss_step,
                        "train_loss": train_loss_step,}
                    )