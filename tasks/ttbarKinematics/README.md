# Reconstruct the $t\bar{t}$ kinematics

> # TODO
> 
> 1. - [ ] **Set the CMS Plot Style**
> 
> 2. Reconstruct the variables:
>    
>    - [x] $M\_t$
>    
>    - [x] $M\_{\bar{t}}$
>    
>    - [x] $M\_{W^+}$
>    
>    - [x] $M\_{W^-}$
>    
>    - [x] Divide all in leptonic and hadronic decays
>    
>    - [ ] fit a Breit-Wigner for each histogram (this is important to understand the magnitude of the non resonant decay)
> 
> 3. - [x] Do the same plots for $\eta$ and $p_T$
> 
> 4. - [x] Create and histogram containing all the  possible $q\bar{q}$ couples and fill it with the hadronic W decays 
> 
> 5. - [ ] Understand the cuts of montecarlo:
>      
>      https://cms-pdmv.cern.ch/mcm/requests?dataset_name=TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8&page=0&shown=127
>      
>      In the option fragment there is a script in wich there is a path: 
>      
>      /cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/powheg/V2/TT_hvq/TT_hdamp_NNPDF31_NNLO_ljets.tgz
>      
>      Unpack this file on the cernbox (*huge file*) and copy only the useful info on github.
>      
>      (Go to the TWiki and try to understand how to understand how a montecarlo is generated)
>    
>    ##### Optional
>    
>    - [ ] Multithreaded loop over events
> 
> # Fix
> 
> - [ ] For the M plots, create another version with a wider scale on the x axis
> 
> - [ ] Fix the ratio w jet decay (and rename it in W hadronic decay)
> 
> - [ ] Make 2 kind of plots: one divinding for the particle charge (e.g. W+,W-), one dividing for the production (hadronic, leptonic)
> 
> # Questions
> 
> - Why the gluon fusion production (90%) dominate over the quark production (10%)?

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

# LHEPart_pdgId structure

There are 9 particles (the 6th(5) and the 7th(6) are produced by the same W. The same stans for the 8th(7) and 9th(8) ):

* The first 2 (0,1)particles of each events are the incoming particles (gluons)

* The 3rd (2) particle is the additional parton (NLO montecarlo)

* The 4th and the 5th (3,4)are the $q_-$ and $\bar{q}^{'}_+$ produced by  $t\bar{t}$

* Be aware!! The 6th (5) particle could be produced both from the $t$ or the $\bar{t}$. Look at the electric charges
  
  - IF the 6th particle has charge < 0, it comes from a $W^-$
  
  - IF the 6th particle has charge  >0, it comes from a $W^+$

The quark in $t \to q W^\pm$ has the opposite charge of the $W^\pm$ and $t$ has the same charge of the $W$ ( so $t \to q_{-}W^+$  or $\bar{t} \to \bar{q}_{+}W^-$ )

### Observation and doubts

- **The Ws decay only in ud,us,cd,cs pais. The motecarlo was generated with some strange cuts???** 

### Technical doubts

- Ci sono modi più veloci di usare root o va bene usare il plain C in questo modo?
- Esistono degli header preimpostati per importare lo stile dei plot CMS?
- Come posso fare uno scan filtrando per istanza?
  - Semplicemente usa le parentesi quadre [idx]
