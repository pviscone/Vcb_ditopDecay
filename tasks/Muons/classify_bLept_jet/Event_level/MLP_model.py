import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
#from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

# enable gpu if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

cpu = torch.device("cpu")
device = torch.device(dev)


class MLP(torch.nn.Module):

    def __init__(self, hidden_arch=[10, 10], batch_size=1,
                 shuffle=False,
                 x_train=None, y_train=None, x_test=None, y_test=None,
                 event_id_train=None, event_id_test=None,
                 optim={}, early_stopping=None):
        super().__init__()

        self.loss_fn = torch.nn.NLLLoss()
        self.early_stopping = early_stopping

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hidden_arch = hidden_arch

        self.event_id_train = event_id_train
        self.event_id_test = event_id_test

        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)

        self.shuffle_at_each_epoch = shuffle
        self.batch_size = batch_size
        self.n_batch = np.ceil(len(x_train)/batch_size).astype(int)

        self.test_loss = np.array([])
        self.train_loss = np.array([])

        self.n_inputs = x_train.shape[1]
        self.n_outputs = y_train.shape[1]

        #self.layers = torch.jit.trace(torch.nn.ModuleList())
        self.layers = torch.nn.Sequential()
        
        self.layers.append(torch.nn.Linear(self.n_inputs, hidden_arch[0]))

        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)
        
        self.layers.append(torch.nn.ReLU())
        for in_neurons, out_neurons in zip(self.hidden_arch[:-1], self.hidden_arch[1:]):
            self.layers.append(torch.nn.Linear(in_neurons, out_neurons))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(
            self.hidden_arch[-1], int(torch.max(self.y_train).item()+1)))
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)
        self.layers.append(torch.nn.LogSoftmax())

        self.optim_dict = optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)

        self.train_accuracy = []
        self.test_accuracy = []

    def forward(self, x):
        x = (x-self.mean)/(self.std+1e-5)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_loop(self, epochs):
        epoch_loop = tqdm(range(epochs), desc="epoch")

        for epoch in epoch_loop:
            self.train()
            x_train = self.x_train
            y_train = self.y_train
            if self.shuffle_at_each_epoch:
                perm = torch.randperm(self.x_train.size()[0])
                x_train = x_train[perm]
                y_train = y_train[perm]

            # for x_batch, y_batch in self.data_loader:
            for i in range(self.n_batch):
                if ((i+1)*self.batch_size < len(x_train)):
                    x_batch = x_train[i*self.batch_size:(i+1)*self.batch_size]
                    y_batch = y_train[i *
                                      self.batch_size:(i+1)*self.batch_size]
                else:
                    x_batch = x_train[i*self.batch_size:len(x_train)]
                    y_batch = y_train[i*self.batch_size:len(y_train)]
                y_logits = self.forward(x_batch).squeeze()
                train_loss_step = self.loss_fn(y_logits, y_batch.squeeze())
                self.optimizer.zero_grad()
                train_loss_step.backward()
                self.optimizer.step()

            self.eval()
            with torch.inference_mode():

                test_logits = self.forward(self.x_test).squeeze()


                test_loss_step = self.loss_fn(
                    test_logits, self.y_test.squeeze())

                self.test_loss = np.append(
                    self.test_loss, test_loss_step.to(cpu).numpy())

                self.train_loss = np.append(
                    self.train_loss, train_loss_step.to(cpu).numpy())
                
                classified_test=torch.argmax(test_logits,dim=1) == self.y_test.squeeze()
                
                self.test_accuracy.append(classified_test.sum().item()/len(classified_test))
                
                classified_train=torch.argmax(self(self.x_train),dim=1) == self.y_train.squeeze()

                self.train_accuracy.append(classified_train.sum().item()/len(classified_train))
                
            if self.early_stopping != None:
                if not "Patience" in self.early_stopping:
                    raise ValueError(
                        "early_stopping must be a dictionary with key 'Patience'")
                patience = self.early_stopping["Patience"]
                if epoch > patience:
                    if "RMS" in self.early_stopping:
                        if np.std(self.test_loss[-patience:]) < self.early_stopping["RMS"]:
                            print("Early stopping: test loss not changing")
                            break
                    if "Mean" in self.early_stopping and self.early_stopping == True:
                        if np.mean(self.test_loss[-patience:]) > np.mean(self.test_loss[-2*patience:-patience]):
                            print("Early stopping: test loss increasing")
                            break

    def loss_plot(self):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, label="test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        
        plt.subplot(1,2,2)
        plt.plot(self.train_accuracy,label="train")
        plt.plot(self.test_accuracy,label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

    def prob(self, x):
        return torch.exp(self.forward(x))
    
    def classify(self, x):
        return torch.argmax(self.forward(x),dim=1)