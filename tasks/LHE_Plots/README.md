# Reconstruct the $t\bar{t}$ kinematics

> # TODO
> 
> 1. Reconstruct the variables:
>    
>    - [x] $M\_t$
>    
>    - [x] $M\_{\bar{t}}$
>    
>    - [x] $M\_{W^+}$
>    
>    - [x] $M\_{W^-}$
>    
>    - [x] Divide all in leptonic/hadronic and particle/antiparticle decays
>    
>    - [x] fit a Breit-Wigner for each histogram (this is important to understand the magnitude of the non resonant decay)
>    
>    - [x] Do a wider version of the plots to see the tails
> 
> 2. - [x] Do the same plots for $\eta$ and $p_T$
> 
> 3. - [x] Create and histogram containing all the  possible $q\bar{q}$ couples and fill it with the hadronic W decays 
>    
>    ##### Optional
>    
>    - [ ] Multithreaded loop over events (or implement everything using the RDataframes)
> 
> # Questions
> 
> - Why the gluon fusion production (90%) dominate over the quark production (10%)?
>   
>   - At the LHC energy the bjorken variable $x=\frac{|q|^2}{2p_{in}^{\mu} q_{\mu} } $ is small and for small X the gluon PDF is much higher than the quark PDF.
>     
>     For example, at the Tevatron, the CM energy was 1TeV  and ttbar production was mainly from valence quarks

---

# Info

- Object dictionary: [Documentation for mc94X_NANO.root](https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html)
  
  <p align="center">
  <img title="" src=".img/2022-11-22-03-11-37-image.png" alt="" width="510" data-align="center">
  </p>

- pdgId:
  
  <p align="center">
  <img title="" src=".img/2022-11-22-04-28-06-image.png" alt="" width="70" data-align="inline"><img src=".img/2022-11-22-04-28-37-image.png" title="" alt="" width="120">
  </p>

---

# Observation and doubts
