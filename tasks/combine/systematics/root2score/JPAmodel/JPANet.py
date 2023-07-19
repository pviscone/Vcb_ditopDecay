import torch
import numpy as np
from root2score.JPAmodel.Attention_block import Attention
from root2score.JPAmodel.MLP import MLP
from root2score.JPAmodel.SelfAttentionPooling import SelfAttentionPooling
from root2score.JPAmodel.torch_dataset import loader
import gc


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"




class JPANet(torch.nn.Module):

    def __init__(self,
                 mu_arch=None, nu_arch=None, jet_arch=None, event_arch=None, pre_attention_arch=None,
                 jet_attention_arch=None, post_attention_arch=None, final_attention=False,post_pooling_arch=None,
                 masses_arch=None,
                 secondLept_arch=None,
                 n_heads=1,
                 n_jet=7,
                 early_stopping=None, dropout=0.15,device=None):
        super().__init__()
        assert post_attention_arch is not None
        assert post_pooling_arch is not None
        self.device=device
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


        self.secondLept_mlp = MLP(arch=secondLept_arch,
                                  out_activation=torch.nn.SiLU(),
                                  dropout=dropout)
        self.secondLept_norm=torch.nn.LayerNorm(post_attention_arch[-1])


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
        
    def forward(self, mu, nu, jet, masses,secondLept):
        #mu=torch.arctanh((mu-self.mu_mean)/(100*self.mu_std))
        #nu=torch.arctanh((nu-self.nu_mean)/(100*self.nu_std))
        #jet=torch.arctanh((jet-self.jet_mean)/(100*self.jet_std))
        #!Create the pad mask
        pad_mask = ((jet == 0)[:, :, 0].squeeze()).to(torch.float32)
        if pad_mask.dim()==1:
            pad_mask=torch.reshape(pad_mask,(1,pad_mask.shape[0]))
        pad_mask[pad_mask==1]=-torch.inf
        
        #!Normalize the inputs
        mu=(mu-self.mu_mean)/(self.mu_std)
        nu=(nu-self.nu_mean)/(self.nu_std)
        jet=(jet-self.jet_mean)/(self.jet_std)
        masses=((torch.log(1+masses)-self.logmass_mean)/self.logmass_std).squeeze()
            
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
        #attn_mask=pad_mask[:,:,None]+pad_mask[:,None,:]

        jet_mass_mask=self.mass_linear1(out_mass_embed)
        jet_mass_mask=vec_to_sym(jet_mass_mask,self.device)
        #jet_mass_mask[attn_mask]=-torch.inf
        jet_mass_mask=jet_mass_mask.repeat_interleave(self.jet_attention.n_heads,dim=0)
        out_jet = self.jet_attention(out_jet,
                                     attn_mask=jet_mass_mask,key_padding_mask=pad_mask)
        
        #!Concatenate all inputs and total attention +mask
        total_out = torch.cat((out_ev, out_jet), dim=1)
        total_out = self.all_norm(total_out)
        total_out = self.all_preattention_mlp(total_out)
        
        pad_mask = torch.cat((torch.zeros((out_ev.shape[0],1),
                                          device=self.device,
                                          dtype=torch.float32)
                              , pad_mask), dim=1)
        
        
        #total_out = torch.nan_to_num(total_out, nan=0.0)
        #attn_mask=pad_mask[:,:,None]+pad_mask[:,None,:]

        total_mask=self.mass_linear2(out_mass_embed)
        total_mask=vec_to_sym(total_mask,self.device)
        #total_mask[attn_mask]=-torch.inf
        total_mask=total_mask.repeat_interleave(self.all_attention.n_heads,dim=0)
        total_out = self.all_attention(total_out,
                                       attn_mask=total_mask,key_padding_mask=pad_mask)
        
        #!Second Lepton
        
        secondLept=(secondLept)/(self.mu_std)
        secondLept_mask=((secondLept == 0)[:, :, 0].squeeze()).to(torch.float32)
        secondLept_mask[secondLept_mask==1]=-torch.inf
        secondLept_out=self.secondLept_mlp(secondLept)
        secondLept_out=self.secondLept_norm(secondLept_out)
        pad_mask=torch.cat((pad_mask,secondLept_mask),dim=1)
        total_out=torch.cat((total_out,secondLept_out),dim=1)
       
        
        #secondLept_out = torch.nan_to_num(secondLept_out, nan=0.0)


        
        #!Pool and output
        #total_out = torch.nan_to_num(total_out, nan=0.0)
        total_out = self.pooling(total_out,pad_mask=pad_mask)
        total_out = self.post_pooling_mlp(total_out)
        total_out = self.output(total_out)

        return total_out

    
    def predict(self,dataset,bunch=1):
        self.eval()
        with torch.inference_mode():
            bunch_size=int(dataset.data["Lepton"].shape[0]//bunch)
            res=torch.zeros((1,3),device=self.device,dtype=torch.float32)
            for mu_bunch,nu_bunch,jet_bunch,mass_bunch,secondLept_bunch in loader(bunch_size,dataset.data["Lepton"],dataset.data["MET"],dataset.data["Jet"],dataset.data["Masses"],dataset.data["SecondLept"]):
                mu_bunch = mu_bunch.to(self.device,non_blocking=True)
                nu_bunch = nu_bunch.to(self.device,non_blocking=True)
                jet_bunch = jet_bunch.to(self.device,non_blocking=True)
                secondLept_bunch = secondLept_bunch.to(self.device,non_blocking=True)
                mass_bunch = mass_bunch.to(self.device,non_blocking=True).squeeze()
                temp_res=self.forward(mu_bunch,nu_bunch,jet_bunch,mass_bunch,secondLept_bunch)
                res=torch.cat((res,temp_res),dim=0)
        
        del mu_bunch,nu_bunch,jet_bunch,mass_bunch,secondLept_bunch
        gc.collect()
        torch.cuda.empty_cache()
        return res[1:]
            

        
def vec_to_sym(matrix,device):
    n=int((np.sqrt(1+8*matrix.shape[1])-1)/2)
    tri_idx=torch.triu_indices(n,n)
    z=torch.zeros(matrix.shape[0],n,n,device=device,dtype=torch.float32)
    z[:,tri_idx[0],tri_idx[1]]=matrix
    return z+torch.transpose(z,2,1)-torch.diag_embed(torch.diagonal(z,dim1=2,dim2=1))



# To change the output dim change the last layer and the res tensor in predict