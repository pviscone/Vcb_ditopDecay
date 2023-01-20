#!/usr/bin/env python
import os
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import *

# this takes care of converting the input files from CRAB
from PhysicsTools.NanoAODTools.postprocessing.framework.crabhelper import inputFiles, runsAndLumis

from PhysicsTools.NanoAODTools.postprocessing.examples.exampleModule import *
p = PostProcessor(".",
                  inputFiles(),
                  "LHEPart_pdgId[7]==-4 && LHEPart_pdgId[6]==5",
                  provenance=True,
                  fwkJobReport=True)
p.run()

print("DONE")
