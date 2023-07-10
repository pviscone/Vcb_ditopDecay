#rm -f report.txt
#rm -f ../../../../Preselection_Skim/{signal/signal,powheg/TTSemiLept,diHad/TTdiHad,diLept/TTdiLept,WJets/WJets}*_MuonCuts.root
export rootfilePATH="/scratchnvme/pviscone/Preselection_Skim"
rm -f $rootfilePATH/NN/**/*.root




echo "------------------- signal Muon --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/signal/signal_train.root/","../../../../Preselection_Skim/NN/train/root/signal_train_MuonCuts.root")'
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/signal/signal_test.root/","../../../../Preselection_Skim/NN/test/root/signal_test_MuonCuts.root")'
ln -s ../../../../Preselection_Skim/NN/test/root/signal_test_MuonCuts.root ../../../../Preselection_Skim/NN/predict/root/signal_predict_MuonCuts.root
echo ""


echo "----------------- signal Electrons -----------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/signal_Electrons/BigElectrons.root/","../../../../Preselection_Skim/NN/predict/root/signal_Electrons_predict_MuonCuts.root")'


echo ""
echo "-------------------- signal Tau --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/signal_Tau/signal_tau.root/","../../../../Preselection_Skim/NN/predict/root/signal_Taus_predict_MuonCuts.root")'

echo ""
echo "-------------------- Semilept --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/powheg/root_files/train","../../../../Preselection_Skim/NN/train/root/TTSemiLept_train_MuonCuts.root")'
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/powheg/root_files/test","../../../../Preselection_Skim/NN/test/root/TTSemiLept_test_MuonCuts.root")'
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/powheg/root_files/predict","../../../../Preselection_Skim/NN/predict/root/TTSemiLept_predict_MuonCuts.root")'
echo ""
echo "---------------------- diHad --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diHad/root_files/","../../../../Preselection_Skim/NN/predict/root/TTdiHad_predict_MuonCuts.root")'
echo ""
echo "---------------------- diLept --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diLept/root_files/train","../../../../Preselection_Skim/NN/train/root/TTdiLept_train_MuonCuts.root")'
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diLept/root_files/test","../../../../Preselection_Skim/NN/test/root/TTdiLept_test_MuonCuts.root")'
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diLept/root_files/predict","../../../../Preselection_Skim/NN/predict/root/TTdiLept_predict_MuonCuts.root")'
echo ""
echo "---------------------- WJets --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/WJets/root_files/","../../../../Preselection_Skim/NN/predict/root/WJets_predict_MuonCuts.root")'
