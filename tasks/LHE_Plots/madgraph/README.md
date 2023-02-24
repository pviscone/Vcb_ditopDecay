# Reconstruct the $t\bar{t}$ kinematics/madgraphMLM dataset

> # TODO
> 
> Do the same plot that are in [Reconstruct the t\bar{t} kinematics](../README.md)
> 
> Add the following plots:
> 
> 1. For the single particles ($b,\bar{b},...$)
>    
>    - [x] $\eta$
>    
>    - [x] $p_t$
> 
> 2. For the couples $(b,\bar{b}$  from tops $),(u,c),(d,b,s)bar$ from Ws) and for the couples (l,q):
>    
>    - [x] $\Delta \phi$
>    
>    - [x] $\Delta \eta$
>    
>    - [x] $\Delta R = \sqrt{\Delta\phi^2+\Delta\eta^2}$
> 
> 3. - [x] Plot the leading $\eta$ and $p_t$ for the quarks from W and make an histogram of the leading type of quarks for both
> 
> 4. - [x] Plot $\eta$ ordered in $p_{t}$ and $p_t$ ordered in $\eta$
> 
> 5. - [x] Plot the $\Delta \eta, \Delta \phi, \Delta R$ for all possible couple (b,bbar,l,q,qbar)
> 
> 6. - [x] Plot the $\Delta R_{min}$ and the histogram for each particle vs the others (5 hists) and to a stackhist with the particles
> 
> 7. - [x] Plot the absoltute $\Delta R_{min}$ and the relative histogram for the 10 possible couple 
> 
> 8. Compute the branching ratio for the $W\to ub$ decays
> 
> 9. Look at the MC cuts in the gridpack
> 
> 10. - [x] Implement all with the RDataframe
> 
> 11. - [x] Add the mean and the rms in the legend
> 
> # FIXME
> 
> - Fix the labels and specify q up qbar antidown 
> 
> - Try to uniform the style of the plots
> 
> - Try to understand why in the ordered pt histogram the bbars are harder than the b 

--- 

# Info

- dataset: [/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&instance=prod/global)
- file: [/store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/73B85577-0234-814E-947E-7DCFC1275886.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL18NanoAODv9%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FNANOAODSIM%2F106X_upgrade2018_realistic_v16_L1v1-v1%2F2560000%2F73B85577-0234-814E-947E-7DCFC1275886.root&instance=prod/global) (2.9GB)

---

---

# LHEPart_pdgId structure

- 0,1: incoming particles

- 2: b from $t$

- 3,4 $l\nu$ form the $W^+$  (2,3,4 from $t$)

- 5 $\bar{b}$ from $\bar{t}$ 

- 6,7 $l\nu$ from $W^-$  (5,6,7 from $\bar{t}$ )

- Additional partons (up to 3)

# Gridpack

- MC portal: https://cms-pdmv.cern.ch/mcm/requests?produce=%2FTTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1%2FNANOAODSIM&page=0&shown=127
  
  Go to the root of the chain and show the fragment. The path to the gridpack is
  
  ```bash
  /cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/madgraph/V5_2.7.3/tt0123j_1l_tbar_5f_ckm_LO_MLM/tt0123j_1l_tbar_5f_ckm_LO_MLM_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz
  ```

# Observation and doubts

- The branching ratio for $W \to ub$ on all hadronic W decays from tt is: $ \frac{|V_{ub}|^2}{|V_{ud}|^2+|V_{us}|^2+|V_{ub}|^2+|V_{cd}|^2+|V_{cs}|^2+|V_{cb}|^2   } \simeq 7 \cdot 10^{-6}$ 

- The tops are excluded for mass constraint
