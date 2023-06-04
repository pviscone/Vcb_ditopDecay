
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
#train_dataset.slice(0,500000)
#test_dataset.slice(0,int(500000/4))

signal_mask=test_dataset.data["label"].squeeze()==1
bkg_mask=test_dataset.data["label"].squeeze()==0

importlib.reload(significance)
def show_significance(mod,
                      func=lambda x: np.arctanh(x),
                      bins=np.linspace(0,7,100),
                      normalize="lumi",
                      ratio_log=True,
                      log=True,
                      bunch=7,
                      **kwargs):
    score=torch.exp(mod.predict(test_dataset,bunch=bunch)[:,-1])
    signal_score=score[signal_mask].detach().to(cpu).numpy()
    bkg_score=score[bkg_mask].detach().to(cpu).numpy()
    #bkg_score=torch.exp(model.predict(powheg_dataset,bunch=150)[:,-1])

    significance.significance_plot(func(signal_score),
                                func(bkg_score),
                                bins=bins,
                                normalize=normalize,
                                ratio_log=ratio_log,
                                log=log,
                                **kwargs)
    plt.show()
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
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               )
model=torch.compile(model)
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
                 show_each=1,
                 train_bunch=20,
                 test_bunch=20/4,
                 batch_size=20000,
                 loss=torch.nn.NLLLoss(weight=torch.tensor([0.25,1.]).to(device)),
                 optim={"lr": 1e-3, "weight_decay": 0.00, },
                 callback=show_significance,
                 shuffle=True,
                 )

#!---------------------Plot loss---------------------
model.loss_plot()
# model.graph(test_dataset)



# %%
#!---------------------Plot significance---------------------
show_significance(model,
                func=lambda x: np.arctanh(x),
                normalize="lumi",
                bins=np.linspace(0,6.5,50),
                ylim=(1e-4,1e8),
                ratio_log=True,
                log=True,
                bunch=8)
# %%
