





# [Skimmed_background](./Skimmed_background)

**This is the background dataset used in the analysis**

Skimming script in[CBOnlySemileptonicFilter/selectBackground](../tasks/CBOnlySemileptonicFilter/selectBackground/README.md) (leptons were balanced)

3 files (in results):

- TTbarSemileptonic_Nocb_optimized.root

> optimized = merged optimizing the tree basket size 

Dataset obtained skimming (using RDataframe) NON cb events from the files (in input_files):

- input_files/leptonFromT.root [/store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2520000/FDB5DCAF-D98E-9C4D-9057-9E11BBEA6970.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL18NanoAODv9%2FTTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8%2FNANOAODSIM%2F106X_upgrade2018_realistic_v16_L1v1-v1%2F2520000%2FFDB5DCAF-D98E-9C4D-9057-9E11BBEA6970.root&instance=prod/global)

- input_files/leptonFromTBar.root [/store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/1A14B4BF-6F04-0D4F-8841-313AEA3804E2.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL18NanoAODv9%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FNANOAODSIM%2F106X_upgrade2018_realistic_v16_L1v1-v1%2F2560000%2F1A14B4BF-6F04-0D4F-8841-313AEA3804E2.root&instance=prod/global)

# [Generated signal](./generated_signal)

Dataset generated using FULLSIM (scripts in [generateCbEvents](../tasks/generateCbEvents/README.md) )

**This is the signal dataset used for the analysis**

3 files:

- TTbarSemileptonic_cbOnlyFromT_pruned.root

- TTbarSemileptonic_cbOnlyFromTbar_pruned.root

- TTbarSemileptonic_cbOnly_pruned_optimized.root

> pruned = ParameterSets tree removed (It was weirdly huge and useless)
> 
> optimized = merging optimizing the tree basket size

# [Muons](./Muons)

Dataset generated skimming the signal and the background dataset with [some cuts on muon events](../tasks/Muons/README.md)

file:

- TTbarSemileptonic_Nocb_optimized_MuonSelection.root

- TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root

**In them there is only the Events TTree**

---

---

---

# Not used in the analysis (bad folder)

**Dataset used only in the first stage of the LHE analysis. Not used for anything else**

## [LHE_Plots/madgraph](./bad/LHE_Plots/madgraph)

file:

73B85577-0234-814E-947E-7DCFC1275886.root

- dataset: [/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&instance=prod/global)

- file: [/store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/73B85577-0234-814E-947E-7DCFC1275886.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL18NanoAODv9%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FNANOAODSIM%2F106X_upgrade2018_realistic_v16_L1v1-v1%2F2560000%2F73B85577-0234-814E-947E-7DCFC1275886.root&instance=prod/global) (2.9GB)
  
  **Observation**: cb events have always the same lepton

## [LHE_Plots/poweg](./bad/LHE_Plots/poweg)

file:

file ttbar.root: [A761E638-9C89-644F-8C33-801D58DEB328.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL17NanoAODv2%2FTTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8%2FNANOAODSIM%2F106X_mc2017_realistic_v8-v1%2F120000%2FA761E638-9C89-644F-8C33-801D58DEB328.root&instance=prod/global)

## [Skimmed_signal_bad](./bad/Skimmed_signal_bad)

(NOT USED)

Skimming scripts in [CBOnlySemileptonicFilter/NANOAODTools](../tasks/CBOnlySemileptonicFilter/NANOAODTools/README.md)

3 files:

- cbFromT_SingleLeptFromTbar.root

- cbFromTbar_SingleLeptFromT.root

- TTbar_Semileptonic_cbOnly.root (the two above merged)

Dataset obtained skimming (using NanoAODTools on CRAB) cb events from the datasets:

- cbFromT_SingleLeptFromTbar: [/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?instance=prod/global&input=file+dataset%3D%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM)
- cbFromTbar_SingleLeptFromT: [/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FTTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&instance=prod/global)

**Observation**: cb events have always the same lepton
