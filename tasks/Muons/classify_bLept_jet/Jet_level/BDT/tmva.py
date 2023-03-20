#%%
import ROOT
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# %%
#!------------------Data loading-------------------


df = pd.read_pickle("../Jet_features.pkl", compression="bz2")

#df = df.drop(columns=["Jet_mass", "max_dEta_Jets","min_dEta_Jets", "min_dPhi_Jets", "max_dPhi_Jets", "dEta_Jet_nu", "dPhi_Jet_nu", "Jet_btag"])

columns = ["Jet_CvBtag", "dPhi_Jet_mu", "Jet_pt", "dEta_Jet_mu",
           "dPhi_Jet_nu", "min_dEta_Jets", "T_mass", "Jet_eta", "label"]
df=df[columns]


# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
columns.remove("label")

test_size = 0.5

train_data, test_data, train_label, test_label = train_test_split(
    data_df, label, test_size=test_size, shuffle=False)
#%%

sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=False, cmap='viridis')

X_train = train_data.to_numpy()
y_train = train_label[:,1]
X_test = test_data.to_numpy()
y_test = test_label[:,1]



# %%


#!Cannot use mask. ROOT complains
"""
X_train_signal = (X_train[train_label[:, 1] == 1])
X_train_background = (X_train[train_label[:, 1] == 0])
 """
X_train_signal = X_train
X_train_background = X_train

root_file = ROOT.TFile("my_file.root", "RECREATE")
signal_tree = ROOT.TTree("signal", "Signal")
branches = []
for idx,col in enumerate(columns):
    branch_name = f"{col}"
    var_array = X_train_signal[:,idx]
    branches.append((branch_name, var_array))
    signal_tree.Branch(branch_name, var_array, f"{branch_name}/F")
for i in range(X_train_signal.shape[0]):
    for j in range(X_train_signal.shape[1]):
        branches[j][1][0] = X_train_signal[i][j]
    signal_tree.Fill()


background_tree = ROOT.TTree("background", "Background")
branches = []
for idx, col in enumerate(columns):
    branch_name = f"{col}"
    var_array = X_train_background[:, idx]
    branches.append((branch_name, var_array))
    background_tree.Branch(branch_name, var_array, f"{branch_name}/F")
for i in range(X_train_background.shape[0]):
    for j in range(X_train_background.shape[1]):
        branches[j][1][0] = X_train_background[i][j]
    background_tree.Fill()







#%%
ROOT.TMVA.Tools.Instance()



# Define the variables
factory = ROOT.TMVA.Factory("TMVAClassification","!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification")

# Create a TMVA data loader and add variables
dataloader = ROOT.TMVA.DataLoader("dataset")
for col in columns:
    dataloader.AddVariable(f"{col}", 'F')
dataloader.AddSignalTree(signal_tree, 1.0)
dataloader.AddBackgroundTree(background_tree, 1.0)
dataloader.PrepareTrainingAndTestTree(
    "", "", "SplitMode=Random:NormMode=NumEvents:!V")

# Create a TMVA factory and train a BDT
factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT",
                   "!H:!V:NTrees=100:MinNodeSize=2.5%:MaxDepth=3")
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
root_file.Write()
root_file.Close()
# %%

#! This not works
gui = ROOT.TMVA.TMVAGui(root_file.GetName())
gui.ShowROCCurve(factory)
gui.PrintResults(factory)

# %%
