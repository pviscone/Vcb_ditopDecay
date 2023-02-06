> # TODO
> 
> - Create the madgraph cards
> 
> - Build the gridpack
> 
> - Generate LHE files
> 
> - Generate events
> 
> ## Info
> 
>  https://cms-pdmv.cern.ch/mcm/campaigns

The tutorial about event generation is [here](../../tutorials/EventGeneration.md)

# Job reports

- [cbFromT Grafana](https://monit-grafana.cern.ch/d/cmsTMDetail/cms-task-monitoring-task-view?orgId=11&var-user=pviscone&var-task=230201_182456%3Apviscone_crab_TTbarSemileptonic_cbOnlyFromT_TuneCP5_13TeV-madgraphMLM-pythia8_FULLSIM&from=1675272296000&to=now)

- [cbFromTbar Grafana](https://monit-grafana.cern.ch/d/cmsTMDetail/cms-task-monitoring-task-view?orgId=11&var-user=pviscone&var-task=230201_182559%3Apviscone_crab_TTbarSemileptonic_cbOnlyFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8_FULLSIM&from=1675272360000&to=now)

# How to run it

1. Make sure that you have all the CMSSW version needed (in the table below) in ~/home/scratch0

2. Make sure that you have added to CMSSW_10_6_30_patch1 (initial version):
   
   - The fragments in Configuration/GenProduction/python
   - run_generic_tarball_xrootd.sh in GenInterface/LHEInterface/data

3. scram b CMSSW_10_6_30_patch1

4. run ./cmsDriver_StepByStep.sh to generate the configuration files

5. run crab submit -c crab_cfg.py (edit the job names if you want)

6. Merge the output files with haddnano.py

**NB:** I don't know why but the NANO job does not prune the ParameterSets TTree. Use the script RemoveParameterSets.c to remove the TTree and then use hadd to shrink it.



Maybe, this can be resolved with an additional step (NANOAODEDMSIM) before the NANOAODSIM step

## Campaign

**RunIISummer20UL18**

## Step vs CMSSW version

| Step                            | CMSSW          |
| ------------------------------- | -------------- |
| LHE,GEN                         | 10_6_30_patch1 |
| SIM                             | 10_6_17_patch1 |
| DIGI,DATAMIX,L1,DIGI2RAW        | 10_6_17_patch1 |
| HLT:2018v32                     | 10_2_16_UL     |
| RAW2DIGI,L1Reco,RECO,RECOSIM,EI | 10_6_17_patch1 |
| PAT                             | 10_6_20        |
| NANO                            | 10_6_26        |

## Xrootd

I used run_generic_tarball_xrootd.sh in the fragment to read my gridpack 

- root://eosuse.cern.ch//eos/user/p/pviscone/Vcb_ditopDecay/tasks/generateCbEvents/Cards/cbFromT/ttbar_semilept_cbOnlyFromT_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz
- root://eosuse.cern.ch//eos/user/p/pviscone/Vcb_ditopDecay/tasks/generateCbEvents/Cards/cbFromTbar/ttbar_semilept_cbOnlyFromTbar_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz

To run this script, CMSSW_10_6_30_patch1 need two shared libraries: 

- libnsl.so.2

- libtirpc.so.3

You have to insert them in the input files and add them to PATH and LD_LIBRARY_PATH
