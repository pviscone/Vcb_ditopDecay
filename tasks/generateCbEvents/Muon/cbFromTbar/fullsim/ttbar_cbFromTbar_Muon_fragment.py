import FWCore.ParameterSet.Config as cms

gridpackPath="root://eosuser.cern.ch//eos/user/p/pviscone/root_files/gridpack/cbFromTbar_Muon_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz"

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring(gridpackPath),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    generateConcurrently = cms.untracked.bool(True),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_xrootd.sh')
)
import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi  import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *
generator = cms.EDFilter("Pythia8ConcurrentHadronizerFilter",
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         PythiaParameters = cms.PSet(
                            pythia8CommonSettingsBlock,
                             pythia8CP5SettingsBlock,
                             pythia8PSweightsSettingsBlock,
                             JetMatchingParameters = cms.vstring(
                                 'JetMatching:setMad = off',
                                 'JetMatching:scheme = 1',
                                 'JetMatching:merge = on',
                                 'JetMatching:jetAlgorithm = 2',
                                 'JetMatching:etaJetMax = 5.',
                                 'JetMatching:coneRadius = 1.',
                                 'JetMatching:slowJetPower = 1',
                                 'JetMatching:qCut = 70.', #this is the actual merging scale
                                 'JetMatching:nQmatch = 5', #4 corresponds to 4-flavour scheme (no matching of b-quarks), 5 for 5-flavour scheme
                                 'JetMatching:nJetMax = 3', #number of partons in born matrix element for highest multiplicity
                                 'JetMatching:doShowerKt = off', #off for MLM matching, turn on for shower-kT matching
                             ),
                             processParameters = cms.vstring(
                                 'TimeShower:mMaxGamma = 1.0',#cutting off lepton-pair production 
                                 ##in the electromagnetic shower to not overlap with ttZ/gamma* samples
                             ),                                
                             parameterSets = cms.vstring('pythia8CommonSettings',
                                                         'pythia8CP5Settings',
                                                         'JetMatchingParameters','pythia8PSweightsSettings','processParameters'
                             )
                         )
)



# Link to generator fragment:
# genFragments/Hadronizer/top.py
