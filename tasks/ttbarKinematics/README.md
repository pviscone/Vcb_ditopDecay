# Reconstruct the $t\bar{t}$ kinematics

> # TODO
> 
> 1. - [ ] **Set the CMS Plot Style** (Partially done (?))
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
>    - [x] fit a Breit-Wigner for each histogram (this is important to understand the magnitude of the non resonant decay)
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
>    - [ ] Multithreaded loop over events (or implement everything using the RDataframes)
> 
> # Fix
> 
> - [x] For the M plots, create another version with a wider scale on the x axis
> 
> - [x] Fix the ratio w jet decay (and rename it in W hadronic decay)
> 
> - [x] Make 2 kind of plots: one divinding for the particle charge (e.g. W+,W-), one dividing for the production (hadronic, leptonic)
> 
> # Questions
> 
> - Why the gluon fusion production (90%) dominate over the quark production (10%)?
>   
>   - At the LHC energy the bjorken variable $x=\frac{|q|^2}{2p_{in}^{\mu} q_{\mu} } $ is small and for small X the gluon PDF is much higher than the quark PDF.
>     
>     For example, at the Tevatron, the CM energy was 1TeV  and ttbar production was mainly from valence quarks

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

- **The Ws decay only in ud,us,cd,cs pais. The motecarlo was generated with some strange cuts???** :
  
  Probably only the Cabibbo mixing is enabled.
  
  In TT_hdamp_NNPDF31_NNLO_ljets/poweg.input there is:
  
  ```fortran
  topdecaymode 11111   ! an integer of 5 digits that are either 0, or 2, representing in 
                       ! the order the maximum number of the following particles(antiparticles)
                       ! in the final state: e  mu tau up charm
                       ! For example
                       ! 22222    All decays (up to 2 units of everything)
                       ! 20000    both top go into b l nu (with the appropriate signs)
                       ! 10011    one top goes into electron (or positron), the other into (any) hadrons,
                       !          or one top goes into charm, the other into up
                       ! 00022    Fully hadronic
                       ! 00002    Fully hadronic with two charms
                       ! 00011    Fully hadronic with a single charm
                       ! 00012    Fully hadronic with at least one charm
  
  semileptonic 1      ! uncomment if you want to filter out only semileptonic events. For example,
                       ! with topdecaymode 10011 and semileptonic 1 you get only events with one top going
                       ! to an electron or positron, and the other into any hadron.
  
  ! Parameters for the generation of spin correlations in t tbar decays
  tdec/wmass 80.4  ! W mass for top decay
  tdec/wwidth 2.141
  tdec/bmass 4.8
  tdec/twidth  1.31 ! 1.33 using PDG LO formula
  tdec/elbranching 0.108
  tdec/emass 0.00051
  tdec/mumass 0.1057
  tdec/taumass 1.777
  tdec/dmass   0.100
  tdec/umass   0.100
  tdec/smass   0.200
  tdec/cmass   1.5
  tdec/sin2cabibbo 0.051
  ```

```bash
runcmsgrid.sh:process="hvq"
runcmsgrid_par.sh:process="hvq"
```

Manual powheg-hvq (see page 6 and 7):

https://mobydick.mib.infn.it/~nason/POWHEG/HeavyQuarks/Powheg-hvq-manual-1.01.pdf

### Technical doubts

- Ci sono modi più veloci di usare root o va bene usare il plain C in questo modo? Conviene imparare a usare rdataframe per sfruttare multithreading implicito?
- 
- Come posso fare uno scan filtrando per istanza?
  - Semplicemente usa le parentesi quadre [idx]
