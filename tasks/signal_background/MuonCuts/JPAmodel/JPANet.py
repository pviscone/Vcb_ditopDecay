import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from Attention_block import Attention
from MLP import MLP
from torchview import draw_graph
from SelfAttentionPooling import SelfAttentionPooling
from livelossplot import PlotLosses
from torch_dataset import loader
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
                 masses_arch=None,
                 n_heads=1,
                 n_jet=7,
                 early_stopping=None, dropout=0.15):
        super().__init__()
        assert post_attention_arch is not None
        assert post_pooling_arch is not None

        self.liveloss = PlotLosses(
            groups={'Loss': ['train_loss', 'test_loss'], })
        self.log={}
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

        self.mass_mlp = MLP(arch=masses_arch,out_activation=torch.nn.SiLU(),dropout=dropout)
        self.mass_norm = torch.nn.LayerNorm(masses_arch[-1])
        self.mass_linear1 = torch.nn.Linear(masses_arch[-1],int(n_jet*(n_jet+1)/2))
        self.mass_linear2 = torch.nn.Linear(masses_arch[-1],int((n_jet+1)*(n_jet+2)/2))
        
        self.jet_mlp = MLP(arch=jet_arch,out_activation=torch.nn.SiLU(),dropout=dropout)


        
        if jet_attention_arch is not None:
            self.jet_attention = Attention(input_dim=jet_arch[-1],
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

        self.output=MLP(arch=[post_pooling_arch[-1],3],out_activation=torch.nn.LogSoftmax(dim=1),
                        dropout=None)




        self.epoch=0
        self.jet_mean=torch.tensor([6.3964e1,-4.7165e-3,-2.3042e-3,3.6918e-1,5.0398e-1,4.2594e-1],device=device)
        self.jet_std=torch.tensor([50.1006,1.8142,1.6456,0.4087,0.3379,0.3796],device=device)
        self.mu_mean=torch.tensor([6.5681e1,5.0793e-5,-1.7228e-3],device=device)
        self.mu_std=torch.tensor([38.2836,1.8144,1.12],device=device)
        self.nu_mean=torch.tensor([66.7863,0.1410,0.7417],device=device)
        self.nu_std=torch.tensor([66.7836,1.7177,1.2575],device=device)
        self.logmass_mean=torch.tensor([4.512]+[5.4]*7+[2.277]+[4.857]*6+[2.277]+[4.857]*5+[2.277]+[4.857]*4+[2.277]+[4.857]*3+[2.277]+[4.857]*2+[2.277]+[4.857]+[2.277],device=device)
        self.logmass_std=torch.tensor([0.2378]+[0.483]*7+[0.431]+[0.7387]*6+[0.431]+[0.7387]*5+[0.431]+[0.7387]*4+[0.431]+[0.7387]*3+[0.431]+[0.7387]*2+[0.431]+[0.7387]+[0.431],device=device)
        
    def forward(self, mu, nu, jet, masses):
        #mu=torch.arctanh((mu-self.mu_mean)/(100*self.mu_std))
        #nu=torch.arctanh((nu-self.nu_mean)/(100*self.nu_std))
        #jet=torch.arctanh((jet-self.jet_mean)/(100*self.jet_std))
        
        #!Normalize the inputs
        pad_mask = ((jet == 0)[:, :, 0].squeeze()).to(torch.bool)
        mu=(mu-self.mu_mean)/(self.mu_std)
        nu=(nu-self.nu_mean)/(self.nu_std)
        jet=(jet-self.jet_mean)/(self.jet_std)
        masses=((torch.log(1+masses)-self.logmass_mean)/self.logmass_std).squeeze()
        
        #!Create the pad mask
        pad_mask = ((jet == 0)[:, :, 0].squeeze()).to(torch.bool)
        if pad_mask.dim()==1:
            pad_mask=torch.reshape(pad_mask,(1,pad_mask.shape[0]))
            
        #!W inputs
        out_mu = self.mu_mlp(mu)
        out_nu = self.nu_mlp(nu)
        out_ev = self.ev_mlp(torch.cat((out_mu, out_nu), dim=2))

        del out_mu, out_nu

        #!Jet inputs
        out_jet = self.jet_mlp(jet)
        
        #!Mass inputs
        out_mass_embed=self.mass_mlp(masses)
        out_mass_embed=self.mass_norm(out_mass_embed)
        
        #! Jet attention + mask
        attn_mask=pad_mask[:,:,None]+pad_mask[:,None,:]

        jet_mass_mask=self.mass_linear1(out_mass_embed)
        jet_mass_mask=vec_to_sym(jet_mass_mask)
        jet_mass_mask[attn_mask]=-torch.inf
        jet_mass_mask=jet_mass_mask.repeat_interleave(self.jet_attention.n_heads,dim=0)
        out_jet = self.jet_attention(out_jet,attn_mask=jet_mass_mask)
        
        #!Concatenate all inputs and total attention +mask
        total_out = torch.cat((out_ev, out_jet), dim=1)
        total_out = self.all_norm(total_out)
        total_out = self.all_preattention_mlp(total_out)
        
        pad_mask = torch.cat((torch.tensor(np.zeros(
                                            out_ev.shape[0]),
                                    device=device, dtype=torch.bool)
                            .unsqueeze(1), pad_mask), dim=1)
        
        
        total_out = torch.nan_to_num(total_out, nan=0.0)
        attn_mask=pad_mask[:,:,None]+pad_mask[:,None,:]

        total_mask=self.mass_linear2(out_mass_embed)
        total_mask=vec_to_sym(total_mask)
        total_mask[attn_mask]=-torch.inf
        total_mask=total_mask.repeat_interleave(self.all_attention.n_heads,dim=0)
        total_out = self.all_attention(total_out,attn_mask=total_mask)
        
        #!Pool and output
        total_out = torch.nan_to_num(total_out, nan=0.0)
        total_out = self.pooling(total_out,pad_mask=pad_mask)
        total_out = self.post_pooling_mlp(total_out)
        total_out = self.output(total_out)

        return total_out

    def train_loop(self, train,test,epochs,train_bunch=1,test_bunch=1,batch_size=1,show_each=False,optim={},loss=None,callback=None,shuffle=False,save_each=None):
        self.optim_dict = optim
        self.optimizer = torch.optim.RAdam(self.parameters(), **self.optim_dict)
        assert loss is not None
        epoch_loop = tqdm(range(epochs), desc="epoch")
        torch.backends.cudnn.benchmark = True
        out = Output()
        display(out)
        bunch_size = int(np.ceil(len(train)/train_bunch))
        assert bunch_size>batch_size
        self.loss_fn=loss
        for epoch in epoch_loop:
            if shuffle:
                train.shuffle()
            
            self.epoch+=1
            self.train()
            temp_train_loss=[]

            for mu_bunch,nu_bunch,jet_bunch,mass_bunch,y_bunch in (loader(bunch_size,train.data["Lepton"],train.data["MET"],train.data["Jet"],train.data["Masses"],train.data["label"])):

                mu_bunch = mu_bunch.to(device,non_blocking=True)
                nu_bunch = nu_bunch.to(device,non_blocking=True)
                jet_bunch = jet_bunch.to(device,non_blocking=True)
                mass_bunch = mass_bunch.to(device,non_blocking=True).squeeze()
                
                y_bunch =y_bunch.to(torch.long).to(device,non_blocking=True)
        
                n_batch=int(np.ceil(y_bunch.shape[0]/batch_size))
                for n in range(n_batch):
                    mu_batch = mu_bunch[n*batch_size:(n+1)*batch_size]
                    nu_batch = nu_bunch[n*batch_size:(n+1)*batch_size]
                    jet_batch = jet_bunch[n*batch_size:(n+1)*batch_size]
                    mass_batch = mass_bunch[n*batch_size:(n+1)*batch_size]
                    y_batch = y_bunch[n*batch_size:(n+1)*batch_size]
                    y_logits = self.forward(
                        mu_batch, nu_batch, jet_batch,mass_batch)
                    
                    train_loss_step = self.loss_fn(
                        y_logits, y_batch.squeeze())
                    
                    self.optimizer.zero_grad()
                    train_loss_step.backward()
                    self.optimizer.step()
                    temp_train_loss.append(train_loss_step.to(cpu).detach().item())


                
                    

            self.eval()
            with torch.inference_mode():
                y_test=test.data["label"].to(torch.long).to(device,non_blocking=True)
                test_logits = self.predict(
                    test,bunch=test_bunch)

                test_loss_step = self.loss_fn(
                    test_logits, y_test.squeeze())


                self.test_loss = np.append(
                    self.test_loss, test_loss_step.to(cpu).numpy())

                self.train_loss = np.append(
                    self.train_loss,np.mean(np.array(temp_train_loss)))


                    
                    
                    
                if(show_each):

                    self.log["train_loss"]=self.train_loss[-1]
                    self.log["test_loss"]=self.test_loss[-1]
                    with out:
                        self.liveloss.update(self.log)
                        if(epoch%show_each==0):
                            self.liveloss.send()
                            self.liveloss.draw()
                            if callback is not None:
                                callback(self)
                if save_each is not None:
                    if(self.epoch%save_each==0 and self.epoch!=0):
                        torch.save(self.state_dict(),f"state_dict_{self.epoch}.pt")
                            

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
        plt.figure()
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, label="test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("loss")



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
            bunch_size=int(dataset.data["label"].shape[0]//bunch)
            res=torch.zeros((1,3),device=device,dtype=torch.float32)
            for mu_bunch,nu_bunch,jet_bunch,mass_bunch,y_bunch in loader(bunch_size,dataset.data["Lepton"],dataset.data["MET"],dataset.data["Jet"],dataset.data["Masses"],dataset.data["label"]):
                mu_bunch = mu_bunch.to(device,non_blocking=True)
                nu_bunch = nu_bunch.to(device,non_blocking=True)
                jet_bunch = jet_bunch.to(device,non_blocking=True)
                mass_bunch = mass_bunch.to(device,non_blocking=True).squeeze()
                y_bunch =y_bunch.to(torch.long).to(device,non_blocking=True)
                temp_res=self.forward(mu_bunch,nu_bunch,jet_bunch,mass_bunch)
                res=torch.cat((res,temp_res),dim=0)
        return res[1:]
            

        
def vec_to_sym(matrix):
    n=int((np.sqrt(1+8*matrix.shape[1])-1)/2)
    tri_idx=torch.triu_indices(n,n)
    z=torch.zeros(matrix.shape[0],n,n,device=device,dtype=torch.float32)
    z[:,tri_idx[0],tri_idx[1]]=matrix
    return z+torch.transpose(z,2,1)-torch.diag_embed(torch.diagonal(z,dim1=2,dim2=1))



# To change the output dim change the last layer and the res tensor in predict