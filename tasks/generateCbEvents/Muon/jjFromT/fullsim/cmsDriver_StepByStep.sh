#! /bin/bash

# step 1: LHE,GEN
# step 2:

#------------PARAMETERS-----------------

#Common parameters
fragment="Configuration/GenProduction/python/ttbar_jjFromT_Muon_fragment.py"
customize="Configuration/DataProcessing/Utils.addMonitoring"
conditions="106X_upgrade2018_realistic_v11_L1v1" 
beamspot="Realistic25ns13TeVEarly2018Collision"
geometry="DB:Extended"
era="Run2_2018"
n="2000"

#Pileup and premix parameters
pileup="dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL18_106X_upgrade2018_realistic_v11_L1v1-v2/PREMIX"
procModifiers="premix_stage2"
datamix="PreMix"

#HLT:2018v32
HLTConditions="102X_upgrade2018_realistic_v15"


#MINIAOD
procModifiersMini="run2_miniAOD_UL"
conditionsMini="106X_upgrade2018_realistic_v16_L1v1"

#NANOAOD
eraNano="run2_nanoAOD_106Xv2"
conditionsNANO="106X_upgrade2018_realistic_v16_L1v1"




#---------------STEP BY STEP-----------------------
#-------------------------------------------------

# Start with 10_6_30_patch1 (Add run_generic_tarball_xrootd.sh and fragment)

here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_30_patch1/src
cmsenv
cd $here

cp ./*fragment.py $CMSSW_BASE/src/Configuration/GenProduction/python
cd $CMSSW_BASE/src
scram b
cd $here

#-----------STEP 1: LHE:GEN -------------------


cmsDriver.py $fragment --python_filename step1.py --eventcontent RAWSIM,LHE --customise $customize --datatier GEN,LHE --fileout file:step1.root --conditions $conditions --beamspot $beamspot --step LHE,GEN --geometry $geometry --era $era --no_exec --mc -n $n

cat RandomSeed.py >> step1.py


# CMSSW_10_6_17_patch1
here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_17_patch1/src
cmsenv
cd $here

#-----------STEP 2: SIM -----------------------

cmsDriver.py --python_filename step2.py --eventcontent RAWSIM --customise $customize --datatier GEN-SIM --fileout file:step2.root --conditions $conditions --beamspot $beamspot --step SIM --geometry $geometry --filein file:step1.root --era $era --runUnscheduled --no_exec --mc -n $n

cat RandomSeed.py >> step2.py

#-----------STEP 3: DIGI,DATAMIX,L1,DIGI2RAW----

cmsDriver.py --python_filename step3.py --eventcontent PREMIXRAW --customise $customize  --datatier GEN-SIM-DIGI --fileout file:step3.root --pileup_input $pileup --conditions $conditions --step DIGI,DATAMIX,L1,DIGI2RAW --datamix $datamix --procModifiers $procModifiers  --geometry $geometry --filein file:step2.root --era $era --runUnscheduled --no_exec --mc -n $n

cat RandomSeed.py >> step3.py


# CMSSW_10_2_16_UL

here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_2_16_UL/src
cmsenv
cd $here

#----------STEP 4: HLT--------------------------

cmsDriver.py --python_filename step4.py --eventcontent RAWSIM --customise $customize --datatier GEN-SIM-RAW --fileout file:step4.root --conditions $HLTConditions --step HLT:2018v32 --filein file:step3.root --era $era --no_exec --mc -n $n --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True)'


cat RandomSeed.py >> step4.py

# CMSSW_10_6_17_patch1

here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_17_patch1/src
cmsenv
cd $here


#---------STEP 5: RAW2DIGI,L1Reco,RECO,RECOSIM,EI--

cmsDriver.py --python_filename step5.py --eventcontent AODSIM --customise $customize --datatier AODSIM --fileout file:step5.root --conditions $conditions --step RAW2DIGI,L1Reco,RECO,RECOSIM,EI --geometry $geometry --filein file:step4.root --era $era --runUnscheduled --no_exec --mc -n $n


cat RandomSeed.py >> step5.py

# CMSSW_10_6_20


here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_20/src
cmsenv
cd $here


#--------STEP 6: MINIAOD --------------------------

cmsDriver.py --python_filename step6.py --eventcontent MINIAODSIM --customise $customize --datatier MINIAODSIM --fileout file:step6.root --conditions $conditionsMini --step PAT --procModifiers $procModifiersMini --geometry $geometry --filein file:step5.root --era $era --runUnscheduled --no_exec --mc -n $n --nThreads 4


cat RandomSeed.py >> step6.py
# CMSSW_10_6_26


here=`pwd`
cd $CMSSW_BASE/..
cd CMSSW_10_6_26/src
cmsenv
cd $here



#-------STEP 7: NANOAOD ---------------------------

cmsDriver.py step1 --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 106X_upgrade2018_realistic_v16_L1v1 --step NANO --nThreads 2 --era Run2_2018,run2_nanoAOD_106Xv2  --filein file:step6.root --fileout file:step7.root -n $n --python_filename step7.py --no_exec


cat RandomSeed.py >> step7.py
