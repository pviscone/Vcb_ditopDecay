# Reconstruct the $t\bar{t}$ kinematics/madgraphMLM dataset

> # TODO
> 
> Do the same plot that are in [Reconstruct the t\bar{t} kinematics](../README.md)
> 
> Add the following plots:
> 
> 1. For the single particles ($b,\bar{b},...$)
>    
>    - [ ] $\eta$
>    
>    - [ ] $p_t$
> 
> 2. For the couples $(b,\bar{b}$  from tops $),(u,d,b$ from Ws) and for the couples (l,q):
>    
>    - [ ] $\Delta \phi$
>    
>    - [ ] $\Delta \eta$
>    
>    - [ ] $\Delta R = \sqrt{\Delta\phi^2+\Delta\eta^2}$
> 
> 3. Compute the branching ratio for the $W\to ub$ decays

--- 

# Info

- dataset: [/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&instance=prod/global)
- file: [0BCB1429-8B19-3245-92C4-68B3DD50AC78.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL18NanoAODv9%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FNANOAODSIM%2F106X_upgrade2018_realistic_v16_L1v1-v1%2F2560000%2F0BCB1429-8B19-3245-92C4-68B3DD50AC78.root&instance=prod/global) (1.6GB)

---

---

# LHEPart_pdgId structure

- 0,1: incoming particles

- 2: b from $t$

- 3,4 $l\nu$ form the $W^+$  (2,3,4 from $t$)

- 5 $\bar{b}$ from $\bar{t}$ 

- 6,7 $l\nu$ from $W^-$  (5,6,7 from $\bar{t}$ )

- Additional partons (up to 3)

# Observation and doubts
