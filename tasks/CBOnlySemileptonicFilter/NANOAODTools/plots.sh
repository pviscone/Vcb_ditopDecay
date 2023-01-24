#! /bin/bash
root -b 'Plots.cpp("./cbFromT/cbFromT_SingleLeptFromTbar.root","TTJets_SingleLeptFromTbar_TuneCP5_13TeV (Only cb)","./cbFromT/images")'
root -b 'Plots.cpp("./cbFromTbar/cbFromTbar_SingleLeptFromT.root","TTJets_SingleLeptFromT_TuneCP5_13TeV (Only cb)","./cbFromTbar/images")'
root -b 'Plots.cpp("./TTbar_Semileptonic_cbOnly.root","TTJets_SingleLept_TuneCP5_13TeV (Only cb)","./images")'
