#%% Imports
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tqdm import tqdm

import MLP_model
import importlib
importlib.reload(MLP_model)
MLP=MLP_model.MLP
OrderedDict=MLP_model.OrderedDict

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("./Jet_features.pkl", compression="bz2")

# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
#%% #! -------------- test/train data-------------------

train_data_df, test_data_df, train_label, test_label = train_test_split(
    data_df, label, test_size=0.2,shuffle=False)

event_id_train=train_data_df["event_id"]
event_id_test=test_data_df["event_id"]

train_label = torch.tensor(train_label, device=device, dtype=torch.float32)
test_label = torch.tensor(test_label, device=device, dtype=torch.float32)

train_data_df=train_data_df.loc[:, train_data_df.columns != "event_id"]
test_data_df=test_data_df.loc[:, test_data_df.columns != "event_id"]


def train(train_data, test_data, train_label, test_label):
  #!---------------------To tensor----------------------------
  train_data = torch.tensor(train_data.values, device=device)
  test_data = torch.tensor(test_data.values, device=device)

  #!---------------------Model definition---------------------
  model=MLP(x_train=train_data,y_train=train_label,x_test=test_data,y_test=test_label,
            event_id_test=event_id_test,
            hidden_arch=[40,40,40],batch_size=30000,
            optim={"lr":0.002,"weight_decay":0.0001,}
            )
  model=model.to(device)

  #!---------------------Training---------------------
  model.train_loop(epochs=1000)

  #!---------------------Plot loss---------------------
  model.loss_plot()
  
  #!------------------Efficiency on events-------------------
  global efficiency
  efficiency=model.evaluate_on_events()
  print(f"Efficiency on events: {efficiency}")
  return model

#%% #!------------------Train-------------------
model=train(train_data_df, test_data_df, train_label, test_label)
all_efficiency=efficiency
print(f"Efficiency on events (all features): {efficiency}")

#model.train_loop(epochs=300)
#model.loss_plot()

# %% #!------------------N-1-------------------

n_minus1_efficiency=[]
col_removed=[]


#TODO Show the intermediate plots
n_col_loop=tqdm(range(len(train_data_df.columns)-1), desc="N-1 loop")
for n_col in n_col_loop:
  if n_col>0:
      train_data_n_1=train_data_n_1.loc[:, train_data_n_1.columns != col_to_remove]
      test_data_n_1=test_data_n_1.loc[:, test_data_n_1.columns != col_to_remove]
  else:
    train_data_n_1=train_data_df
    test_data_n_1=test_data_df
  
  dict_n_minus1 = OrderedDict({})
  column_loop=tqdm(train_data_n_1.columns, desc="Column")
  for col in column_loop:
      train_data_n_1_step = train_data_n_1.loc[:, train_data_n_1.columns != col]
      test_data_n_1_step = test_data_n_1.loc[:, test_data_n_1.columns != col]
      model=train(train_data_n_1_step, test_data_n_1_step, train_label, test_label)
      plt.title(f"N-{n_col+1}: {col}")
      plt.show()
      dict_n_minus1[col] = efficiency

  dict_n_minus1 = dict_n_minus1.sort()
  col_to_remove = dict_n_minus1[-1][0]
  col_removed.append(f"N={n_col+1}: {col_to_remove}")
  n_minus1_efficiency.append(dict_n_minus1[-1][1])
# %%


col_removed.insert(0,"All features")
n_minus1_efficiency.insert(0,all_efficiency)

plt.barh(np.arange(len(col_removed)), n_minus1_efficiency,
         height=0.5, tick_label=col_removed, xerr=1/np.sqrt(len(test_data_df.groupby(level=0))), ecolor="black", color="orange", capsize=2)
plt.xlabel("Efficiency")



#TODO: Do the N+1 thing