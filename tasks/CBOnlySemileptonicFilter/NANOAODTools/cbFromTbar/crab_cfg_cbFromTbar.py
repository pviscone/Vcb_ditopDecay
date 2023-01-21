from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config

config = Configuration()

config.section_("General")
config.General.requestName = "VcbSkimming_SingleLeptFromT"
config.General.transferLogs = True
config.section_("JobType")
config.JobType.pluginName = "Analysis"
config.JobType.psetName = "PSet.py"
config.JobType.scriptExe = "crab_script_cbFromTbar.sh"
# hadd nano will not be needed once nano tools are in cmssw
config.JobType.inputFiles = ["crab_script_cbFromTbar.py", "haddnano.py"]
config.JobType.sendPythonFolder = True
config.section_("Data")
config.Data.inputDataset = "/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"
# config.Data.inputDBS = 'phys03'
config.Data.inputDBS = "global"
config.Data.splitting = "FileBased"
# config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = 30
# config.Data.totalUnits = 58178650

config.Data.outLFNDirBase = "/store/user/pviscone"
config.Data.publication = False
config.Data.outputDatasetTag = "NanoTestPost"
config.section_("Site")
config.Site.storageSite = "T2_IT_Pisa"

# config.Site.storageSite = "T2_CH_CERN"
# config.section_("User")
# config.User.voGroup = 'dcms'
