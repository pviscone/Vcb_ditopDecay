#%%
#!------------------Imports-------------------

import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn as sk
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.naive_bayes import GaussianNB

# %%
#!------------------Data loading-------------------


df = pd.read_pickle("./Jet_features.pkl", compression="bz2")

#df = df.drop(columns=["Jet_mass", "max_dEta_Jets","min_dEta_Jets", "min_dPhi_Jets", "max_dPhi_Jets", "dEta_Jet_nu", "dPhi_Jet_nu", "Jet_btag"])

df = df[["Jet_CvBtag", "dPhi_Jet_mu", "Jet_pt", "dEta_Jet_mu", "dPhi_Jet_nu",
         "min_dEta_Jets", "T_mass", "Jet_eta", "Jet_phi", "label", "event_id"]]


# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]

test_size = 0.25


event_id = data_df["event_id"]
data_df = data_df.loc[:, data_df.columns != "event_id"]
_, event_id_test, = train_test_split(
    event_id, test_size=test_size, shuffle=False)


train_data, test_data, train_label, test_label = train_test_split(
    data_df, label, test_size=test_size, shuffle=False)


X_train = train_data.to_numpy()
y_train = train_label[:, 1]
X_test = test_data.to_numpy()
y_test = test_label[:, 1]

#%%




algos={"Fisher":LinearDiscriminantAnalysis(),
        "knn-500": sk.neighbors.KNeighborsClassifier(n_neighbors=1000, n_jobs=-1),
         }



#algos={}

res = test_data.copy()
res["label"] = test_label[:, 1]
efficiency={}

for algo in algos:
    algos[algo].fit(X_train,y_train)
    #! WARNING: class_weight don't exist in official sklearn. You have to implement it yourself
    y_prob=algos[algo].predict_proba(X_test)[:,1]
    y_pred=algos[algo].predict(X_test)
    print(algo)
    
    #!Note: don0t consider it for knn. Predict does not have class_weights
    print(classification_report(y_test,y_pred))
    print("")
    
    res[f"{algo}_score"] = y_prob
    idx = res.groupby(level=0)[f"{algo}_score"].idxmax()
    eff = res["label"][idx].sum()/len(idx)
    print(f"{algo} Efficiency: {eff}")
    efficiency[algo]=eff

# %%
