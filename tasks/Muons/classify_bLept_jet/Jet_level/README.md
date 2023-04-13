# Dataset

After the preselection, the following cuts were applied:

- dR LHE-Jet<0.4

- Partons that match to the same jet => Event removed

- Just the first 7 jet in pt are considered (if one parton match a discarded jet, the event is discarded)

The dataset is a dataframe in which each row is a jet and the features are:

```python
["Jet_mass","Jet_btag","max_dEta_Jets","Jet_eta","max_dPhi_Jets","min_dPhi_Jets","Jet_CvL_tag",
"dEta_Jet_nu","T_mass","min_dEta_Jets","dPhi_Jet_nu","dEta_Jet_mu","Jet_pt","dPhi_Jet_mu","Jet_CvB"]
```

We have to assign the bLept parton to the right Jet.

Each jet is classified and a score is assigned to the jet. At the end the event is reconstructed picking the highest score in the event

# MLP

A simple multilayer perceptron of size [input_dim, 40, 40, 40, 2]

(validation size=0.2)

|              | Efficiency on Events |
| ------------ | -------------------- |
| ttbar_cbOnly | $\sim$ 67%           |
| BigMuons     | $\sim 70$%           |

TRY ON LXPLUS-GPU WITH A BIGGER NETWORK for BIGMUONS

### N-1, N+1

With this model we did the N+1,N-1 procedure to rank the feature.

We start with all the features and repeat the training removing one feature in rotation.

Then we see what is the most "useless" feature and we discard it.

Then we repeat the procedure with the remaining feature.

<img title="" src=".img/d1c0a67365899e72fe56f90a18847d7486768f98.png" alt="all_steps.png" width="477">

The first 8 feature are selected

(N+1 is  in agreement with the N-1 results)

# Fisher

Test size=0.25

(for both ttbar_cbOnly and BigMuons dataset)

Efficiency on events: 45%

# KNN

Test_size=0.25

Only on ttbar_cbOnly. BigMuons require too much ram (if you want, try on cmsanalysis)

k=1000 (k=500 is the same)

Efficiency on events = 55.4%

# BDT

On TMVA

```python
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


method = factory.BookMethod(dataloader,ROOT.TMVA.Types.kBDT, "BDTG", "!H:!V:NTrees=200:MinNodeSize=5%:\
BoostType=Grad:Shrinkage=.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=3:")
```

Only on ttbar_cbOnly.

Efficiency on events 65%
