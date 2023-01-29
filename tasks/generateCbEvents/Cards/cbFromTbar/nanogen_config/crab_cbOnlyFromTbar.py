from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'TTbarSemileptonic_cbOnlyFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8_NANOGEN'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'PrivateMC'
config.JobType.psetName = './cbOnlyFromTbar_nanogen.py'
config.JobType.allowUndistributedCMSSW = True
config.JobType.numCores = 1

config.Data.outputPrimaryDataset = 'TTbarSemileptonic_cbOnlyFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8_max3j_NANOGEN'
config.Data.splitting = 'EventBased'
config.Data.unitsPerJob = 1000
NJOBS = 500  # This is not a configuration parameter, but an auxiliary variable that we use in the next line.
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.outLFNDirBase = '/store/user/pviscone/'
config.Data.publication = False
config.Data.outputDatasetTag = 'TTbar_Semileptonic_cbOnlyFromTbar_NANOGEN'

config.Site.storageSite = 'T2_IT_Pisa'
