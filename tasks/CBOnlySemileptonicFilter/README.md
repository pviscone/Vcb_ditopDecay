> # TODO
> 
> Generate a pure $ t\bar{t} \to bbcbl\nu_l$ dataset filtering the previous TTTosemileptonic dataset  (both singleLeptFromT and TBar)
> 
> - Do it first with a simple root macro
> - Filter an existing dataset
> 
> ## Datasets
> 
> - cbFromT_SingleLeptFromTbar: [/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?instance=prod/global&input=file+dataset%3D%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM)
> - cbFromTbar_SingleLeptFromT: [/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FTTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&instance=prod/global)

## Info

- [NanoAODTools skimming tutorial](../../tutorials/Skimming.md)

---

## RDataFrame

A simple root script to filter the data in local using RDataFrame

## NANOAOD Tools (on CRAB)

[GitHub - cms-nanoAOD/nanoAOD-tools: Tools for working with NanoAOD (requiring only python + root, not CMSSW)](https://github.com/cms-nanoAOD/nanoAOD-tools)

Scripts to run the skimming on crab using nanoaod tools 

- "*_cbFrom_Tbar" files refers to cbFromTbar_SingleLeptFromT dataset

- The files without a suffix refers to cbFromT_SingleLeptFromTbar dataset

---

---

# Observations

- The skimmed dataset has:
  
  - $\sim$ 30k entries for the SingleLeptFromTbar case over 58178650 events
  
  - $\sim$ 15k entries for the SingleLeptFromT case over 57904914 events
  
  Also looking to the CRAB logs, the preselection fraction is $\sim$ 0.05% vs 0.03%
  
  The total BR for the semileptonic cb over all the semileptoni is 0.039%
  
  **WHY????**
