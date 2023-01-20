# Skimming NANOAODs

To skim NANOAODs the most feasible way is to use NANOAOD-Tools

 [GitHub - cms-nanoAOD/nanoAOD-tools: Tools for working with NanoAOD (requiring only python + root, not CMSSW)](https://github.com/cms-nanoAOD/nanoAOD-tools)

## Build NanoAOD Tools

In CMSSW_xx_x_x/src

```bash
git clone https://github.com/cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools
cd PhysicsTools/NanoAODTools
cmsenv
scram b
cd $CMSSW_BASE/src
cmsenv
```

## Local skimming on lxplus

```python
#skimming.py
import ROOT 
from importlib import import_module

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

#Create a List [] with filenames as strings
files=["root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/73B85577-0234-814E-947E-7DCFC1275886.root"]

#Create a postprocessor, passing it the necessary and optional parameters
#Try running with justcount=True, and modifying the numbers in the cut=" " parameter string.
#Then, switch the justcount=False, and when it runs, it will produce a skim of NanoAOD with that number of events.
#Try opening the file in the command line with 'root -l 06CC0D9B-4244-E811-8B62-485B39897212_CH01-Skim.root'
#At the ROOT prompt, use 'new TBrowser' to get a GUI 
p=PostProcessor(".", #This tells the postprocessor where to write the ROOT files
                files,
                cut="LHEPart_pdgId[3]==4 && LHEPart_pdgId[4]==5",
                modules=[],
                #jsonInput={1 : [[10000, 10010]]}, #This json input limits the postprocessor to only events in run 1, and lumesections 10000 to 10010
                postfix="_CH01-Skim", #This will be attached to the end of the filename if it's output
                #justcount=True, #When True, just counts events that pass the cut=" " and jsonInput criteria. When False, will let this produce a skim!
                )

#Up to this point, things have only been imported and defined. 
#Nothing gets processed until the postprocessor we've created (and named "p") has its run command invoked:
p.run()
```

```bash
python skimming.py
```

(Modules defines new branches)

## Skimming on the GRID using CRAB

In order to run it on crab you need to configure 3 files:

(Example to skim some NANOAOD files)

File: crab_cfg.py (crab configuration file)

[CRAB3ConfigurationFile](https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3ConfigurationFile)

```python
from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config

config = Configuration()

config.section_("General")
config.General.requestName = 'VcbSkimming_SingleLeptFromTbar_noAuto'
config.General.transferLogs = True
config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'PSet.py'
config.JobType.scriptExe = 'crab_script.sh'
# hadd nano will not be needed once nano tools are in cmssw
config.JobType.inputFiles = ['crab_script.py','haddnano.py']
config.JobType.sendPythonFolder = True
config.section_("Data")
config.Data.inputDataset = '/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM'
#config.Data.inputDBS = 'phys03'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
#config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = 93
#config.Data.totalUnits = 58178650

config.Data.outLFNDirBase = '/store/user/pviscone'
config.Data.publication = False
config.Data.outputDatasetTag = 'NanoTestPost'
config.section_("Site")
config.Site.storageSite = "T2_IT_Pisa"
```

file: PSet.py 

I didn't understand what this is meant to do (CMSSW env var stuff)

```python
# this fake PSET is needed for local test and for crab to figure the output
# filename you do not need to edit it unless you want to do a local test using
# a different input file than the one marked below
import FWCore.ParameterSet.Config as cms
process = cms.Process('NANO')
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(),
    # lumisToProcess=cms.untracked.VLuminosityBlockRange("254231:1-254231:24")
)
process.source.fileNames = [
#    '../../NanoAOD/test/lzma.root'  # you can change only this line
]
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.output = cms.OutputModule("PoolOutputModule",
                                  fileName=cms.untracked.string('tree.root'))
process.out = cms.EndPath(process.output)
```

file: crab_script.sh

The exe file that the job will run

```shell
#this is not mean to be run locally
#
echo Check if TTY
if [ "`tty`" != "not a tty" ]; then
  echo "YOU SHOULD NOT RUN THIS IN INTERACTIVE, IT DELETES YOUR LOCAL FILES"
else

echo "ENV..................................."
env 
echo "VOMS"
voms-proxy-info -all
echo "CMSSW BASE, python path, pwd"
echo $CMSSW_BASE 
echo $PYTHON_PATH
echo $PWD 
rm -rf $CMSSW_BASE/lib/
rm -rf $CMSSW_BASE/src/
rm -rf $CMSSW_BASE/module/
rm -rf $CMSSW_BASE/python/
mv lib $CMSSW_BASE/lib
mv src $CMSSW_BASE/src
mv module $CMSSW_BASE/module
mv python $CMSSW_BASE/python

echo Found Proxy in: $X509_USER_PROXY
python crab_script.py $1
fi
```

Run the job with

```shell
crab submit -c crab_cfg.py
```

# T2_IT_Pisa

You will find your things in 

```bash
/gpfs/ddn/srm/cms/store/user/pviscone/
```
