import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from Attention_block import Attention
from MLP import MLP
from torchview import draw_graph
from SelfAttentionPooling import SelfAttentionPooling
from livelossplot import PlotLosses
from dataset import loader
from ipywidgets import Output
from IPython.display import display

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



# enable gpu if available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

cpu = torch.device("cpu")
device = torch.device(dev)


class JPANet(torch.nn.Module):

    def __init__(self,
                 mu_arch=None, nu_arch=None, jet_arch=None, event_arch=None, pre_attention_arch=None,
                 jet_attention_arch=None, post_attention_arch=None, final_attention=False,post_pooling_arch=None,
                 n_heads=1,
                 weight=None,
                 optim={}, early_stopping=None, dropout=0.15):
        super().__init__()
        assert post_attention_arch is not None
        assert post_pooling_arch is not None

        self.liveloss = PlotLosses(
            groups={'Loss': ['train_loss', 'test_loss'],
                    'Acccuracy': ['train_accuracy','test_accuracy'], })
        self.log={}
        
        self.loss_fn = torch.nn.NLLLoss(weight=weight)
        self.early_stopping = early_stopping


        self.test_loss = np.array([])
        self.train_loss = np.array([])

        # Declare the layers here
        if mu_arch is not None:
            self.mu_mlp = MLP(arch=mu_arch,
                              out_activation=torch.nn.SiLU(), dropout=dropout)
        else:
            self.mu_mlp=torch.nn.Identity()

        
        if nu_arch is not None:
            self.nu_mlp = MLP(arch=nu_arch,
                              out_activation=torch.nn.SiLU(), dropout=dropout)

        else:
            self.nu_mlp=torch.nn.Identity()


        self.mu_nu_norm = torch.nn.LayerNorm(event_arch[0])
        
        if event_arch is not None:
            self.ev_mlp = MLP(arch=event_arch,out_activation=torch.nn.SiLU(),
                              dropout=dropout)
        else:
            self.ev_mlp = torch.nn.Identity()

        if jet_arch is not None:
            self.jet_mlp = MLP(arch=jet_arch,out_activation=torch.nn.SiLU(),
                               dropout=dropout)
            attention_input_dim = jet_arch[-1]
        else:
            self.jet_mlp = torch.nn.Identity()
            attention_input_dim = self.jet_train.shape[2]

        
        if jet_attention_arch is not None:
            self.jet_attention = Attention(input_dim=attention_input_dim,
                mlp_arch=jet_attention_arch,n_heads=n_heads, dropout=dropout)
        else:
            self.jet_attention = torch.nn.Identity()
            self.jet_attention.forward = lambda x,key_padding_mask=None: x


        if pre_attention_arch is not None:
            self.all_preattention_mlp = MLP(arch=pre_attention_arch,
                                    out_activation=torch.nn.SiLU(),
                                    dropout=dropout)
        else:
            self.all_preattention_mlp = torch.nn.Identity()

        if final_attention:
            self.all_norm=torch.nn.Identity()
            self.all_attention = Attention(input_dim=post_attention_arch[0], mlp_arch=post_attention_arch[1:],
                                   n_heads=n_heads, dropout=dropout)
        else:
            self.all_norm = torch.nn.LayerNorm(post_attention_arch[0])
            self.all_attention = MLP(arch=post_attention_arch, out_activation=torch.nn.SiLU(),
                             dropout=dropout)
            self.all_attention.forward=lambda x,key_padding_mask=None:self.all_attention.forward(x)
    

    
        self.pooling=SelfAttentionPooling(input_dim=post_attention_arch[-1])
        
        self.post_pooling_mlp=MLP(arch=post_pooling_arch,out_activation=torch.nn.SiLU(),dropout=None)

        self.output=MLP(arch=[post_pooling_arch[-1],2],out_activation=torch.nn.LogSoftmax(dim=1),
                        dropout=None)

        self.optim_dict = optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)

        self.train_accuracy = []
        self.test_accuracy = []


    def forward(self, mu, nu, jet):

        pad_mask = ((jet == -10)[:, :, 0].squeeze()).to(torch.bool)
        if pad_mask.dim()==1:
            pad_mask=torch.reshape(pad_mask,(1,pad_mask.shape[0]))

        out_mu = self.mu_mlp(mu)
        out_nu = self.nu_mlp(nu)
        out_ev = self.ev_mlp(torch.cat((out_mu, out_nu), dim=2))

        del out_mu, out_nu

        out_jet = self.jet_mlp(jet)
        out_jet = self.jet_attention(out_jet,key_padding_mask=pad_mask)
        
        total_out = torch.cat((out_ev, out_jet), dim=1)
        total_out = self.all_norm(total_out)
        total_out = self.all_preattention_mlp(total_out)
        
        pad_mask = torch.cat((torch.tensor(np.zeros(
                                            out_ev.shape[0]),
                                    device=device, dtype=torch.bool)
                            .unsqueeze(1), pad_mask), dim=1)
        total_out = self.all_attention(total_out,key_padding_mask=pad_mask)
        
        total_out = self.pooling(total_out)
        total_out = self.post_pooling_mlp(total_out)
        total_out = self.output(total_out)

        return total_out

    def train_loop(self, train,test,epochs,train_bunch=1,test_bunch=1,batch_size=1,show_each=False):
        epoch_loop = tqdm(range(epochs), desc="epoch")
        torch.backends.cudnn.benchmark = True
        out = Output()
        display(out)
        bunch_size = int(np.ceil(len(train)/train_bunch))
        assert bunch_size>batch_size
        for epoch in epoch_loop:

            self.train()
            temp_train_loss=[]
            temp_train_accuracy=[]

            
            for bunch in loader(train,batch_size=bunch_size):

                mu_bunch,nu_bunch,jet_bunch,y_bunch = bunch
                mu_bunch = mu_bunch.to(device,non_blocking=True)
                nu_bunch = nu_bunch.to(device,non_blocking=True)
                jet_bunch = jet_bunch.to(device,non_blocking=True)
                y_bunch =y_bunch.to(device,non_blocking=True)
        
                n_batch=int(np.ceil(y_bunch.shape[0]/batch_size))
                for n in range(n_batch):
                    mu_batch = mu_bunch[n*batch_size:(n+1)*batch_size]
                    nu_batch = nu_bunch[n*batch_size:(n+1)*batch_size]
                    jet_batch = jet_bunch[n*batch_size:(n+1)*batch_size]
                    y_batch = y_bunch[n*batch_size:(n+1)*batch_size]
                    y_logits = self.forward(
                        mu_batch, nu_batch, jet_batch)
                    
                    train_loss_step = self.loss_fn(
                        y_logits, y_batch.squeeze())
                    
                    self.optimizer.zero_grad()
                    loss=train_loss_step
                    loss.backward()
                    #train_loss_step.backward()
                    self.optimizer.step()
                    
                    temp_train_loss.append(train_loss_step.detach().to(cpu,non_blocking=True).numpy())
                    temp_train_accuracy.append(((torch.argmax(
                        y_logits, axis=1) == y_batch.squeeze()
                    ).sum()/y_batch.shape[0]).detach().to(cpu,non_blocking=True).numpy())
                    del loss,train_loss_step
                
                    

            self.eval()
            with torch.inference_mode():
                y_test=test.label.to(device,non_blocking=True)
                test_logits = self.predict(
                    test,bunch=test_bunch)

                test_loss_step = self.loss_fn(
                    test_logits, y_test.squeeze())


                self.test_loss = np.append(
                    self.test_loss, test_loss_step.to(cpu).numpy())

                self.train_loss = np.append(
                    self.train_loss,np.mean(np.array(temp_train_loss)))

                
                classified_test=(torch.argmax(
                        test_logits, axis=1
                        ) == y_test.squeeze()
                    ).sum()/y_test.shape[0]
                
                self.test_accuracy.append(
                    classified_test.to(cpu).numpy()
                    )
                
                
                """
                #idx=np.random.choice(range(y_train.shape[0]),y_test.shape[0],replace=False)

                classified_train=(torch.argmax(
                        self(mu_train,
                             nu_train,
                             jet_train), axis=1)== y_train.squeeze()
                            ).sum()/y_train.shape[0]
                """

                self.train_accuracy.append(
                    np.mean(np.array(temp_train_accuracy)))
                    
                    
                    
                if(show_each):

                    self.log["train_loss"]=self.train_loss[-1]
                    self.log["test_loss"]=self.test_loss[-1]
                    self.log["test_accuracy"]=self.test_accuracy[-1]
                    self.log["train_accuracy"]=self.train_accuracy[-1]
                    with out:
                        self.liveloss.update(self.log)
                        if(epoch%show_each==0):
                            self.liveloss.send()
                            self.liveloss.draw()
                            

            if self.early_stopping is not None:
                if "Patience" not in self.early_stopping:
                    raise ValueError(
                        "early_stopping must be a dictionary with key 'Patience'")
                patience = self.early_stopping["Patience"]
                if epoch > patience:
                    if "RMS" in self.early_stopping:
                        if np.std(self.test_loss[-patience:]
                                  ) < self.early_stopping["RMS"]:
                            print("Early stopping: test loss not changing")
                            break
                    if "Mean" in self.early_stopping and self.early_stopping is True:
                        if np.mean(self.test_loss[-patience:]
                                   ) > np.mean(self.test_loss[-2*patience:-patience]):
                            print("Early stopping: test loss increasing")
                            break



    def loss_plot(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, label="test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracy, label="train")
        plt.plot(self.test_accuracy, label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

    def graph(self,test_dataset,batch_size=10000):
        mu_test=test_dataset.mu_data.to(device)
        nu_test=test_dataset.nu_data.to(device)
        jet_test=test_dataset.jet_data.to(device)
        model_graph = draw_graph(self, input_data=[
                                 mu_test[:batch_size],
                                 nu_test[:batch_size],
                                 jet_test[:batch_size]],
                                expand_nested=True,depth=5)
        return model_graph.visual_graph

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict(self,dataset,bunch=1):
        self.eval()
        with torch.inference_mode():
            bunch_size=dataset.label.shape[0]//bunch
            res=torch.zeros((1,2),device=device,dtype=torch.float32)
            for bunch in loader(dataset,batch_size=bunch_size):
                mu_bunch,nu_bunch,jet_bunch,y_bunch=bunch
                mu_bunch = mu_bunch.to(device,non_blocking=True)
                nu_bunch = nu_bunch.to(device,non_blocking=True)
                jet_bunch = jet_bunch.to(device,non_blocking=True)
                y_bunch =y_bunch.to(device,non_blocking=True)
                temp_res=self.forward(mu_bunch,nu_bunch,jet_bunch)
                res=torch.cat((res,temp_res),dim=0)
        return res[1:]
            

        