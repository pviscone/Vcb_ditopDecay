from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config

#config = Configuration()


config = config()

config.section_("General")
config.General.transferOutputs = True
config.General.transferLogs = True
config.General.requestName = 'TTbarSemileptonic_cbOnlyFromT_TuneCP5_13TeV-madgraphMLM-pythia8_max3j_FULLSIM'
config.General.workArea = 'crab_projects'

config.section_("JobType")
config.JobType.pluginName = 'PrivateMC'
config.JobType.allowUndistributedCMSSW = True
config.JobType.numCores = 1



config.JobType.psetName = "PSet.py"
config.JobType.scriptExe = "crab_run.sh"
# hadd nano will not be needed once nano tools are in cmssw
config.JobType.inputFiles = ["step1.py", "step2.py","step3.py","step4.py","step5.py","step6.py","step7.py","libnsl.so.2","libtirpc.so.3"]
config.JobType.sendPythonFolder = True



config.section_("Data")

config.Data.outputPrimaryDataset = 'TTbarSemileptonic_cbOnlyFromT_TuneCP5_13TeV-madgraphMLM-pythia8_max3j'
config.Data.splitting = 'EventBased'
config.Data.unitsPerJob = 400
NJOBS = 5000  # This is not a configuration parameter, but an auxiliary variable that we use in the next line.
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.outLFNDirBase = '/store/user/pviscone/'
config.Data.publication = False
config.Data.outputDatasetTag = 'RunIISummer20UL18'


config.section_("Site")
config.Site.storageSite = 'T2_IT_Pisa'
