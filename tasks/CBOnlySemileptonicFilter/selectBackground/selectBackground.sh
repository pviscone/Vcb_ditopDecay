#xrdcp root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/1A14B4BF-6F04-0D4F-8841-313AEA3804E2.root ./leptonFromTbar.root


#xrdcp root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2520000/FDB5DCAF-D98E-9C4D-9057-9E11BBEA6970.root ./leptonFromT.root

root -b 'balanceLeptons.cpp("leptonFromTbar.root","leptonFromTbar")'

hadd -fk leptonFromTbar_new.root *_leptFromTbar.root
root -b 'RemoveEvents.cpp("leptonFromTbar.root")'

hadd -fk TTbarSemileptonic_Nocb_LeptFromTbar.root leptonFromTbar.root leptonFromTbar_new.root


root -b 'balanceLeptons.cpp("leptonFromT.root","leptonFromT")'
hadd -fk leptonFromT_new.root *_leptFromT.root
root -b 'RemoveEvents.cpp("leptonFromT.root")'

hadd -fk TTbarSemileptonic_Nocb_LeptFromT.root leptonFromT.root leptonFromT_new.root


hadd -fk TTbarSemileptonic_Nocb.root TTbarSemileptonic_Nocb_LeptFromTbar.root TTbarSemileptonic_Nocb_LeptFromT.root
