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

## Skimming on the GRID using CRAB
