#rm -f report.txt
#rm -f ../../../../Preselection_Skim/{signal/signal,powheg/TTSemiLept,diHad/TTdiHad,diLept/TTdiLept,WJets/WJets}*_ElectronCuts.root
export rootfilePATH="/scratchnvme/pviscone/Preselection_Skim"
rm -f $rootfilePATH/NN_Electrons/**/*.root




echo "-------------------- signal --------------------"
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/signal_Electrons/signal_Electrons_train.root/","../../../../Preselection_Skim/NN_Electrons/train/root/signal_train_ElectronCuts.root")'
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/signal_Electrons/signal_Electrons_test.root/","../../../../Preselection_Skim/NN_Electrons/test/root/signal_test_ElectronCuts.root")'
ln -s ../../../../Preselection_Skim/NN_Electrons/test/root/signal_test_ElectronCuts.root ../../../../Preselection_Skim/NN_Electrons/predict/root/signal_predict_ElectronCuts.root

echo ""
echo "-------------------- Semilept --------------------"
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/powheg/root_files/train","../../../../Preselection_Skim/NN_Electrons/train/root/TTSemiLept_train_ElectronCuts.root")'
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/powheg/root_files/test","../../../../Preselection_Skim/NN_Electrons/test/root/TTSemiLept_test_ElectronCuts.root")'
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/powheg/root_files/predict","../../../../Preselection_Skim/NN_Electrons/predict/root/TTSemiLept_predict_ElectronCuts.root")'
echo ""
echo "---------------------- diHad --------------------"
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/diHad/root_files/","../../../../Preselection_Skim/NN_Electrons/predict/root/TTdiHad_predict_ElectronCuts.root")'
echo ""
echo "---------------------- diLept --------------------"
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/diLept/root_files/train","../../../../Preselection_Skim/NN_Electrons/train/root/TTdiLept_train_ElectronCuts.root")'
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/diLept/root_files/test","../../../../Preselection_Skim/NN_Electrons/test/root/TTdiLept_test_ElectronCuts.root")'
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/diLept/root_files/predict","../../../../Preselection_Skim/NN_Electrons/predict/root/TTdiLept_predict_ElectronCuts.root")'
echo ""
echo "---------------------- WJets --------------------"
root -l 'ElectronCuts.cpp("../../../../Preselection_Skim/WJets/root_files/","../../../../Preselection_Skim/NN_Electrons/predict/root/WJets_predict_ElectronCuts.root")'
