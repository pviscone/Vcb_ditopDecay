import torch
import numpy as np

#enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
  
device = torch.device(dev)
  
  
class MLP(torch.nn.Module):
    def __init__(self,hidden_arch=[10,10],x_train=None,y_train=None,x_test=None,y_test=None,learning_rate=0.001):
        super().__init__()
        self.optimizer=torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        
        self.loss_fn=torch.nn.BCELoss()
        self.accuracy_fn = lambda y_pred, y_true: (
            y_pred.round() == y_true).sum()/len(y_true)
        
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.arch=hidden_arch

        
        self.test_loss=[]
        self.test_accuracy=[]
        self.train_loss=[]
        self.train_accuracy=[]
        
        self.n_inputs=x_train.shape[1]
        self.n_outputs=y_train.shape[1]
        
        self.layers = []
        self.layers.append(torch.nn.Linear(self.n_inputs, hidden_arch[0]))
        self.layers.append(torch.nn.Sigmoid())
        for in_neurons,out_neurons in zip(self.hidden_arch[:-1],self.hidden_arch[1:]):
            self.layers.append(torch.nn.Linear(in_neurons,out_neurons))
            self.layers.append(torch.nn.Sigmoid())
        
        self.layers.append(torch.nn.Linear(self.hidden_arch[-1],self.n_outputs))
        
        if self.n_outputs==1:
            self.layers.append(torch.nn.Sigmoid())
        else:
            self.layers.append(torch.nn.Softmax())
        
        
        
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def training(self,epochs):
        for epoch in range(epochs):
            self.train()
            
            y_logits=self.forward(self.x_train).squeeze()
            y_pred=y_logits.round()
            
            train_loss_step=self.loss_fn(y_logits,self.y_train)
            train_accuracy_step=self.accuracy_fn(y_pred,self.y_train)
            
            self.optimizer.zero_grad()
            train_loss_step.backward()
            self.optimizer.step()
            
            self.eval()
            with torch.inference_mode():
                test_logits=self.forward(self.x_test).squeeze()
                test_pred=test_logits.round()
                
                test_loss_step=self.loss_fn(test_logits,self.y_test)
                test_accuracy_step=self.accuracy_fn(test_pred,self.y_test)
                
                self.test_loss.append(test_loss_step.numpy())
                self.test_accuracy.append(test_accuracy_step.numpy())
                self.train_loss.append(train_loss_step.numpy())
                self.train_accuracy.append(train_accuracy_step.numpy())
        