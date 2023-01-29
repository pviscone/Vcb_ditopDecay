# How to generate events with madgraph + pythia + cmssw

> **Useful links:**
> 
> - [SWGuideEventGeneration](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEventGeneration)
> 
> - [MadGraph5 Guide](https://twiki.cern.ch/twiki/bin/view/CMS/QuickGuideMadGraph5aMCatNLO)
> 
> - [GenSim Intro](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGenIntro)
> 
> - [SWGuideSimulation](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSimulation)
> 
> - [WorkBookGeneration](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGeneration)
> 
> - [WorkBookSimDigi](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookSimDigi)
> 
> - [CMS Generator Group](https://twiki.cern.ch/twiki/bin/viewauth/CMS/GeneratorMain)
> 
> - [Info for MC production for Ultra Legacy Campaigns 2016, 2017, 2018 - Monte Carlo production tools](https://cms-pdmv.gitbook.io/project/mccontact/info-for-mc-production-for-ultra-legacy-campaigns-2016-2017-2018)
> 
> - [NanoGen](https://twiki.cern.ch/twiki/bin/viewauth/CMS/NanoGen) : 
>   
>   Seems to contain useful informations on how to build NanoAOD from the LHE files using cmsDriver.py

## Generate gridpack

On lxplus/cmsanalysis:

- git clone https://github.com/cms-sw/genproductions.git

- Create the cards (copy some example, the run_cars have standard settings for a specific period of data taking )

- Check if the cards are correctly defined using
  
  ```bash
  cd genproductions/bin/MadGraph5_aMCatNLO/Utilities/parsing_code
  python ./parsing.py ${path to card folder}/${name of cards}
  ```

- To generate the gridpack
  
  ```bash
  cd genproductions/bin/MadGraph5_aMCatNLO
  ./gridpack_generation.sh ${cardsName} ${cardsPath} local
  ```
  
  **NB: to run it you have to clean the CMSSW enviroment**
  
  ```bash
  unset CMSSW_BASE
  ```

## Generate

You have to clone the fragments from genproductions, put in $CMSSW_BASE/src and scram 

```bash
git clone git@github.com:cms-sw/genproductions.git $CMSSW_BASE/src/Configuration/GenProduction/
scram b
```

**You have to create the folder "python" and put inside it the fragments**
