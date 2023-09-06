mkdir -p results
mkdir -p results/root
rm results/*.{root,out,json,pdf} results/root/*
text2workspace.py $1 -o workspace.root --channel-masks | tee results/workspace.out
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2  | tee results/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root results/combine.root
mv workspace.root results/

combineTool.py -M Impacts -d results/workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --expectSignal=1 --points 2000 | tee -a results/combine_log.out
combineTool.py -M Impacts -d results/workspace.root -m 125 --robustFit 1 --doFits --parallel 30 -t -1 --expectSignal=1 --points 2000 | tee -a results/combine_log.out
combineTool.py -M Impacts -d results/workspace.root -m 125 -o results/impacts.json -t -1 --expectSignal=1 --points 2000 | tee -a results/combine_log.out
plotImpacts.py -i results/impacts.json -o results/impact_plot | tee -a results/combine_log.out
mv higgsCombine* results/root/
mv combine_logger.out results/
