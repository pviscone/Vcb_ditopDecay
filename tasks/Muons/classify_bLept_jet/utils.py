import torch
from MLP_model import MLP
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

class OrderedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if type(key) == str:
            return super().__getitem__(key)
        else:
            key = tuple(self.keys())[key]
            return (key, self[key])

    def sort(self):
        return OrderedDict(dict(sorted(self.items(), key=lambda item: item[1])))

    def insert(self,where,pair):
        keys=list(self.keys())
        values=list(self.values())
        keys.insert(where,pair[0])
        values.insert(where,pair[1])
        return OrderedDict(zip(keys,values))
        



def train(df, label, event_id_test, test_size=0.2,
          arch=[40,40,40],batch_size=30000,    optim_kwargs={"lr": 0.002, "weight_decay": 0.0001,}):
    #!---------------------To tensor----------------------------
    train_data,test_data, train_label, test_label=train_test_split(df, label,test_size=test_size,shuffle=False)

    train_data = torch.tensor(train_data.values, device=device)
    test_data = torch.tensor(test_data.values, device=device)
    train_label = torch.tensor(train_label, device=device, dtype=torch.float32)
    test_label = torch.tensor(test_label, device=device, dtype=torch.float32)
    
    assert train_data.dim() == test_data.dim()
    
    if train_data.dim() == 1:
        train_data = train_data.unsqueeze(dim=1)
        test_data = test_data.unsqueeze(dim=1)
    

    #!---------------------Model definition---------------------
    model = MLP(x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label,
                event_id_test=event_id_test,
                hidden_arch=arch, batch_size=batch_size,
                optim=optim_kwargs
                )
    model = model.to(device)

    #!---------------------Training---------------------
    model.train_loop(epochs=1000)

    #!---------------------Plot loss---------------------
    model.loss_plot()

    #!------------------Efficiency on events-------------------
    efficiency = model.evaluate_on_events()
    print(f"Efficiency on events: {efficiency}")
    return efficiency


def plot_efficiency(dict,err):
    keys=list(dict.keys())
    values=list(dict.values())
    arange = np.arange(len(keys))
    all_efficiency = values[0]
    plt.figure(figsize=(7, 5))
    plt.subplots_adjust(left=0.25,right=0.8)
    plt.barh(arange, values, height=0.5, tick_label=keys,
             xerr=err, ecolor="black", color="orange", capsize=2)
    plt.plot([all_efficiency]*2, [-0.5, len(keys)+0.5], c="r")
    plt.fill_between([all_efficiency-err, all_efficiency+err], [-0.5, -0.5],
                     [len(keys)+0.5, len(keys)+0.5], color="grey", alpha=0.5)
    plt.ylim(-0.5, len(keys))
    plt.xlim(np.min(values)/1.2, np.max(values)*1.2)
    plt.xlabel("Efficiency")
    plt.grid(alpha=0.5, linestyle="--")
