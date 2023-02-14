root -b 'muonPreselectionPlot.cpp("../TTbarSemileptonic_Nocb_optimized.root","Nocb","./images/background")'

root -b 'muonPreselectionPlot.cpp("../TTbarSemileptonic_cbOnly_pruned_optimized.root","cbOnly","./images/signal")'

root -b 'muonSelection.cpp("../TTbarSemileptonic_cbOnly_pruned_optimized.root","cbOnly","./images/signal")'

root -b 'muonSelection.cpp("../TTbarSemileptonic_Nocb_optimized.root","Nocb","./images/background")'

root -b 'BtaggingCuts.cpp("./images")'
