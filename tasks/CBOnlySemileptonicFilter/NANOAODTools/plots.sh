#! /bin/bash
root -b 'Plots.cpp("./cbFromT/cbFromT_SingleLeptFromTbar.root","TTJets_cbOnly_SingleLeptFromTbar_TuneCP5_13TeV","./cbFromT/images")'
root -b 'Plots.cpp("./cbFromTbar/cbFromTbar_SingleLeptFromT.root","TTJets_cbOnly_SingleLeptFromT_TuneCP5_13TeV","./cbFromTbar/images")'
root -b 'Plots.cpp("./TTbar_Semileptonic_cbOnly.root","TTJets_cbOnly_SingleLept_TuneCP5_13TeV","./images")'
