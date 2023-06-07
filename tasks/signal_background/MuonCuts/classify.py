
#%%
import sys
sys.path.append("./JPAmodel/")
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import JPAmodel.JPANet as JPA
import JPAmodel.losses as losses
import JPAmodel.significance as significance
JPANet = JPA.JPANet
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cudnn.benchmark = True


if torch.cuda.is_available():
    
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

#!-----------------Load datasets-----------------!#
print("Loading datasets...")
train_dataset=torch.load("../../../root_files/signal_background/Muons/NN/train_Muons.pt")
test_dataset=torch.load("../../../root_files/signal_background/Muons/NN/test_Muons.pt")

#%%
n_diLept_train=torch.sum(train_dataset.data["type"]==-1)
n_signal_train=torch.sum(train_dataset.data["type"]==1)

train_dataset.data["label"][train_dataset.data["type"]==-1]=torch.ones(n_diLept_train).reshape((n_diLept_train,1))
train_dataset.data["label"][train_dataset.data["type"]==1]=2*torch.ones(n_signal_train).reshape((n_signal_train,1))

n_diLept_test=torch.sum(test_dataset.data["type"]==-1)
n_signal_test=torch.sum(test_dataset.data["type"]==1)

test_dataset.data["label"][test_dataset.data["type"]==-1]=torch.ones(n_diLept_test).reshape((n_diLept_test,1))
test_dataset.data["label"][test_dataset.data["type"]==1]=2*torch.ones(n_signal_test).reshape((n_signal_test,1))


#%%
#! Show significance function
#! IT MUST BE DEFINED BEFORE THE TRAINING TO BE USED AS CALLBACK
lumi=138e3
ttbar_1lept=lumi*832*0.44  #all lepton
ttbar_2lept=lumi*832*0.11*0.2365 #all lepton types

signal_mask=test_dataset.data["label"].squeeze()==2
diLept_mask=test_dataset.data["label"].squeeze()==1
semiLept_mask=test_dataset.data["label"].squeeze()==0


had_decay=np.abs(test_dataset.data["HadDecay"][semiLept_mask])
charmed=np.bitwise_or(had_decay[:,0]==4,had_decay[:,1]==4).bool()
charmed=np.bitwise_or(charmed,np.abs(test_dataset.data["AdditionalPartons"][semiLept_mask,0])==4).bool()
up=np.bitwise_or(had_decay[:,0]==2,had_decay[:,1]==2).bool()
def show_significance(mod,
                        func=lambda x: np.arctanh(x),
                        xlim=(0,7),
                        bins=60,
                        log=True,
                        bunch=7,
                        **kwargs):
    score=func(torch.exp(mod.predict(test_dataset,bunch=bunch)[:,-1]).detach().cpu().numpy())
    signal_score=score[signal_mask]
    diLept_score=score[diLept_mask]
    semiLept_score=score[semiLept_mask]
    hist_dict = {
                "signal":{
                    "data":signal_score,
                    "color":"red",
                    "weight":ttbar_1lept*0.517*0.33*(8.4e-4),
                    "histtype":"errorbar",
                    "stack":False,},
                "diLept":{
                    "data":diLept_score,
                    "color":"cornflowerblue",
                    "weight":ttbar_2lept,
                    "stack":True,},
                "SemiLept Up":{
                    "data":semiLept_score[up],
                    "color":"lightsteelblue",
                    "weight":ttbar_1lept*0.179*(1-8.4e-4)*torch.sum(up)/len(up),
                    "stack":True,},
                "SemiLept charm":{
                    "data":semiLept_score[charmed],
                    "color":"plum",
                    "weight":ttbar_1lept*0.179*(1-8.4e-4)*torch.sum(charmed)/len(charmed),
                    "stack":True,},
                
    }
    
    ax1,ax2=significance.make_hist(hist_dict,xlim=xlim,bins=bins,log=log,**kwargs)
# %%
#!---------------------Model---------------------
importlib.reload(JPA)
importlib.reload(losses)
JPANet = JPA.JPANet

mu_feat=3
nu_feat=3
jet_feat=6

model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               masses_arch=[36,128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               n_jet=7,
               )
#model=torch.compile(model)
model = model.to(device)
print(f"Number of parameters: {model.n_parameters()}")

#!---------------------Training---------------------
#model=torch.compile(model)
#torch.save(model.state_dict(), "./state_dict.pt")
#torch.save({"train_loss":model.train_loss,
#            "test_loss":model.test_loss,
#            "epoch":model.epoch}, "./loss.pt")
#state_dict=torch.load("./state_dict_70.pt")
#model.load_state_dict(state_dict)
model.train_loop(train_dataset,test_dataset,
                 epochs=70,
                 show_each=1,
                 train_bunch=20,
                 test_bunch=7,
                 batch_size=20000,
                 loss=torch.nn.NLLLoss(weight=torch.tensor([0.33,1,1.]).to(device)),
                 optim={"lr": 1e-3, "weight_decay": 0.00, },
                 callback=None,
                 shuffle=True,
                 save_each=10
                 )
#callback=show_significance
#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph(test_dataset)



# %%
#!---------------------Plot significance---------------------

show_significance(model)

