# %%
#!------------------Imports-------------------


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import sklearn as sk


# %%
#!------------------Data loading-------------------
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)


df = pd.read_pickle("../Jet_features.pkl", compression="bz2")

#df = df.drop(columns=["Jet_mass", "max_dEta_Jets","min_dEta_Jets", "min_dPhi_Jets", "max_dPhi_Jets", "dEta_Jet_nu", "dPhi_Jet_nu", "Jet_btag"])

df=df[["Jet_CvBtag","dPhi_Jet_mu","Jet_pt","dEta_Jet_mu","dPhi_Jet_nu","min_dEta_Jets","T_mass","Jet_eta","Jet_phi","label","event_id"]]


# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]

test_size = 0.5


event_id = data_df["event_id"]
data_df = data_df.loc[:, data_df.columns != "event_id"]
_, event_id_test, = train_test_split(
    event_id, test_size=test_size, shuffle=False)


train_data, test_data, train_label, test_label = train_test_split(
    data_df, label, test_size=test_size, shuffle=False)
#%%

sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=False, cmap='viridis')

X_train = train_data.to_numpy()
y_train = train_label[:,1]
X_test = test_data.to_numpy()
y_test = test_label[:,1]

dt = DecisionTreeClassifier(max_depth=20,
                            min_samples_leaf=2)
bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=10,
                         learning_rate=0.5)

bdt.fit(X_train, y_train)


y_prob=bdt.predict_proba(X_test)[:,1]

# %%
res=test_data.copy()
res["score"] = y_prob
res["label"] = test_label[:, 1]
idx=res.groupby(level=0)["score"].idxmax()
efficiency = res["label"][idx].sum()/len(idx)
print(f"Efficiency: {efficiency}")
# %%

#plt.figure(dpi=1000)
#sk.tree.plot_tree(bdt[0])

# %%
