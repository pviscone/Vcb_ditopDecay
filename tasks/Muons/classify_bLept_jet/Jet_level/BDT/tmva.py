# %%
import ROOT
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from array import array

# %%
#!------------------Data loading-------------------


df = pd.read_pickle("../Jet_features.pkl", compression="bz2")

#df = df.drop(columns=["Jet_mass", "max_dEta_Jets","min_dEta_Jets", "min_dPhi_Jets", "max_dPhi_Jets", "dEta_Jet_nu", "dPhi_Jet_nu", "Jet_btag"])

columns=["Jet_CvBtag", "dPhi_Jet_mu", "Jet_pt", "dEta_Jet_mu", "dPhi_Jet_nu",
         "min_dEta_Jets", "T_mass", "dEta_Jet_nu", "label"]

df = df[columns]


# [a,b]: a=non leptonic, b=leptonic
label = np.expand_dims(df["label"].astype(float).to_numpy(), axis=1)
ohe = OneHotEncoder()
ohe.fit(label)
label = ohe.transform(label).toarray()

data_df = df.loc[:, df.columns != "label"]
columns.remove("label")

test_size = 0.3

train_data, test_data, train_label, test_label = train_test_split(
    data_df, label, test_size=test_size, shuffle=False)
# %%

sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=False, cmap='viridis')

X_train = train_data.to_numpy()
y_train = train_label[:, 1]
X_test = test_data.to_numpy()
y_test = test_label[:, 1]


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
for idx, col in enumerate(columns):
    branch_name = f"{col}"
    var_array = X_train_signal[:, idx]
    branches.append((branch_name, var_array))
    signal_tree.Branch(branch_name, var_array, f"{branch_name}/F")
for i in range(X_train_signal.shape[0]):
    if (train_label[i, 1] == 1):
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
    if (train_label[i, 1] == 0):
        for j in range(X_train_background.shape[1]):
            branches[j][1][0] = X_train_background[i][j]
        background_tree.Fill()


# %%
ROOT.TMVA.Tools.Instance()


# Define the variables
factory = ROOT.TMVA.Factory("TMVAClassification", root_file,
                            "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification")



# Create a TMVA data loader and add variables
dataloader = ROOT.TMVA.DataLoader("dataset")
for col in columns:
    dataloader.AddVariable(f"{col}", 'F')
dataloader.AddSignalTree(signal_tree, 1.0)
dataloader.AddBackgroundTree(background_tree, 1.0)
dataloader.PrepareTrainingAndTestTree(
    "", "", "SplitMode=Random:NormMode=NumEvents:!V")

# Create a TMVA factory and train a BDT
method = factory.BookMethod(dataloader,ROOT.TMVA.Types.kBDT, "BDTG", "!H:!V:NTrees=200:MinNodeSize=5%:BoostType=Grad:Shrinkage=.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=3:")


"""
ROOT.TMVA.Types.kBDT, "BDT",
"!H:!V:NTrees=100:MinNodeSize=2.5%:MaxDepth=5"

"""


factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
c1 = factory.GetROCCurve(dataloader)
c1.Draw()

# %%
root_file.Write()
root_file.Close()


# Evaluate the BDT

ROOT.TMVA.mvas("dataset", "my_file.root")
for c in ROOT.gROOT.GetListOfCanvases():
    c.Draw()

#gui=ROOT.TMVA.TMVAGui( "my_file.root")


# %%
test_root = ROOT.TFile("test_file.root", "RECREATE")
test_tree = ROOT.TTree("test", "Test")
branches = []
for idx, col in enumerate(columns):
    branch_name = f"{col}"
    var_array = X_test[:, idx]
    branches.append((branch_name, var_array))
    test_tree.Branch(branch_name, var_array, f"{branch_name}/F")
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        branches[j][1][0] = X_test[i][j]
    test_tree.Fill()


# Load the previously trained BDT from an XML file
reader = ROOT.TMVA.Reader("Color:!Silent")

# ... add any other variables ...
branches2 = {}
for col in (columns):
    branches2[col] = array('f', [-999])
    reader.AddVariable(col, branches2[col])
    test_tree.SetBranchAddress(col, branches2[col])

# Book the MVA method
reader.BookMVA("BDT", "./dataset/weights/TMVAClassification_BDTG.weights.xml")

# Loop over the TTree and classify each event
score = []
branch_list = []
exec(f"branch_list=[test_tree.{col} for col in columns]")
for idx, event in enumerate(test_tree):
    test_tree.GetEntry(idx)
    br = []
    for col in columns:
        exec(f"br.append(event.{col})")

    score.append(reader.EvaluateMVA(br, ROOT.TString('BDT')))

    # Do something with the BDT output (e.g., make a histogram)

score = np.array(score)
# %%
working_point = -0.07
test_data["score"] = score
test_data["label"] = test_label[:, 1]
idx_pred = test_data.groupby(level=0)["score"].idxmax()

efficiency = test_data.loc[idx_pred]["label"].sum()/test_data["label"].sum()

# %%
