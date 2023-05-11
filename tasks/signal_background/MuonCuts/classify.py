
#%%
import importlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import mplhep
sys.path.append("./JPAmodel/")
import JPAmodel.JPANet as JPA
from JPAmodel.significance import significance_plot
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
JPANet = JPA.JPANet
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)



train_dataset=torch.load("../../../root_files/signal_background/train_dataset.pt")
test_dataset=torch.load("../../../root_files/signal_background/test_dataset.pt")
""" train_dataset.mu_data=train_dataset.mu_data[:1000000]
train_dataset.nu_data=train_dataset.nu_data[:1000000]
train_dataset.jet_data=train_dataset.jet_data[:1000000]
train_dataset.label=train_dataset.label[:1000000]
test_dataset.mu_data=test_dataset.mu_data[:100000]
test_dataset.nu_data=test_dataset.nu_data[:100000]
test_dataset.jet_data=test_dataset.jet_data[:100000]
test_dataset.label=test_dataset.label[:100000] """

#powheg_dataset=torch.load("../../../root_files/signal_background/powheg_dataset.pt")


#test_dataset.to(device)
#train_dataset.to(device)
#powheg_dataset.to(device)


# %%

importlib.reload(JPA)
JPANet = JPA.JPANet

mu_feat=3
nu_feat=4
jet_feat=8

model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
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
#model.state_dict=torch.load("./state_dict.pt")
model.train_loop(train_dataset,test_dataset,
                 epochs=20,
                 show_each=5,
                 train_bunch=18,
                 test_bunch=8,
                 batch_size=20000,
                 loss=model.NLoss_fn,
                 optim={"lr": 1e-4, "weight_decay": 0.00, })


#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph(test_dataset)


# %%

signal_mask=test_dataset.label.squeeze()==1
bkg_mask=test_dataset.label.squeeze()==0
score=torch.exp(model.predict(test_dataset,bunch=10)[:,-1])
signal_score=score[signal_mask].detach().to(cpu).numpy()
#bkg_score=torch.exp(model.predict(powheg_dataset,bunch=150)[:,-1])
bkg_score=score[bkg_mask].detach().to(cpu).numpy()


func= lambda x: (x)
significance_plot(func(signal_score),
                  func(bkg_score),
                  bins=20,
                  score_range=(0,1),
                  normalize="lumi",
                  log=True)



# %%
