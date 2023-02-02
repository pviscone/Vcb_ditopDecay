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

# EXAMPLE

[generateCbEvents](../tasks/generateCbEvents/README.md)

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

1. Go to https://cms-pdmv.cern.ch/mcm/chained_campaigns and look for the campaign you are intrested in (for each step)
   
   Under the option "sequence"  you will find the cmsDriver.py options that you have to use. **Remember to add -n to defin the number of event**
   
   **Look carefully to the CMSSW used**

2. At the end of each configuration script add
   
   ```python
   from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
   randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
   randHelper.populate()
   ```
   
   This will automatically randomize the seeds

3. Write a bash script in which you cmsenv the right version of CMSSW and cmsRun the step.py

4. Write a dummy PSet.py script that is needed just for staging out the output files

5. Write a crab_cfg.py script in which you define the file to transfer on crab ( config.JobType.inputFiles ) and the bash script to execute ( config.JobType.scriptExe ) 

6. crab submit -c crab_cfg.py (**WITH the initial cmssw version**)

### Appendix: gridpack on EOS

Normally you have to put your gridpack in a cvmfs space and specify in the fragment run_generic_tarball_cvmfs.sh

You can also put your gridpack in eos (root://eosuser.cern.ch) and use the script run_generic_tarball_xrootd.sh

This script exists only in the last versions of the CMSSW software. You can add to the version you are using with:

1. cd $CMSSW_BASE/src

2. git cms-addpkg GeneratorInterface/LHEInterface

3. cd GeneratorInterface/LHEInterface/data

4. wget https://raw.githubusercontent.com/cms-sw/cmssw/master/GeneratorInterface/LHEInterface/data/run_generic_tarball_xrootd.sh

5. scram b

**NOTE:** You have to add some libraries to make this work.

For example, in CMSSW_10_6_30_patch1 you have to add libnsl.so.2 and libtirpc.so.3 to the PATH and LD_LIBRARY_PATH of your  CRAB server.

So basically you have to add these libraries to your input files and define the new paths in the executable script
