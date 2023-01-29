#!/bin/bash
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.sh
scramv1 project CMSSW CMSSW_10_6_29 # cmsrel is an alias not on the workers
ls -alrth
cd CMSSW_10_6_4/src/
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo $CMSSW_BASE "is the CMSSW we created on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
echo "Copying input files to local worker node"
cmsRun cbOnlFromT_nanogen.py
xrdcp -f .cbOnlyFromT_nanogen.root root://eosuser.cern.ch//eos/user/p/pviscone/condorOutput
#rm .cbOnlyFromT_nanogen.root
echo "Job finished on " `date`