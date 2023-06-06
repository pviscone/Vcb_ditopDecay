rm -f report.txt ../../../../Preselection_Skim/{signal/signal_MuonCuts,powheg/TTSemiLept_MuonCuts,diHad/TTdiHad_MuonCuts,diLept/TTdiLept_MuonCuts,WJets/WJets_MuonCuts}.root


echo "-------------------- signal --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/signal/BigMuons.root/","../../../../Preselection_Skim/signal/signal_MuonCuts.root")'
echo ""
echo "-------------------- Semilept --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/powheg/root_files/","../../../../Preselection_Skim/powheg/TTSemiLept_MuonCuts.root")'
echo ""
echo "---------------------- diHad --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diHad/root_files/","../../../../Preselection_Skim/diHad/TTdiHad_MuonCuts.root")'
echo ""
echo "---------------------- diLept --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/diLept/root_files/","../../../../Preselection_Skim/diLept/TTdiLept_MuonCuts.root")'
echo ""
echo "---------------------- WJets --------------------"
root -l 'MuonCuts.cpp("../../../../Preselection_Skim/WJets/root_files/","../../../../Preselection_Skim/WJets/WJets_MuonCuts.root")'
