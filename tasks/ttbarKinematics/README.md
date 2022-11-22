# Reconstruct the $t\bar{t}$ kinematics

> # TODO
> 
> 1. Reconstruct the variables:
>    
>    - [ ] $M\_t$
>    
>    - [ ] $M\_{\bar{t}}$
>    
>    - [ ] $M\_{W^+}$
>    
>    - [ ] $M\_{W^-}$
> 
>         Divide the leptonic and the hadronic decays and fit a Breit-Wigner for each histogram (this is important to understand the magnitude of the non resonant decay)
> 
> 2. - [ ] Do the same plots for $\eta$ and $p_T$
> 3. - [ ] Create and histogram containing all the  possible $q\bar{q}$ couples and fill it with the hadronic W decays 

---

### Info

- Object dictionary: [Documentation for mc94X_NANO.root](https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html)
  
  <p align="center">
  <img title="" src=".img/2022-11-22-03-11-37-image.png" alt="" width="510" data-align="center">
  </p>

- pdgId:
  
  <p align="center">
  <img title="" src=".img/2022-11-22-04-28-06-image.png" alt="" width="70" data-align="inline"><img src=".img/2022-11-22-04-28-37-image.png" title="" alt="" width="120">
  </p>

- file: [A761E638-9C89-644F-8C33-801D58DEB328.root](https://cmsweb.cern.ch/das/request?input=file%3D%2Fstore%2Fmc%2FRunIISummer20UL17NanoAODv2%2FTTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8%2FNANOAODSIM%2F106X_mc2017_realistic_v8-v1%2F120000%2FA761E638-9C89-644F-8C33-801D58DEB328.root&instance=prod/global) (1.6GB) called ttbar.root on the local repo

---

---

# TTree structure

* The first 2 particles of each events are the incoming particles (gluons)
* The 3rd particle is the additional parton (NLO montecarlo)
* Be aware!! The 4th particle could be produced both from the t or the tbar. Look at the electric charges
