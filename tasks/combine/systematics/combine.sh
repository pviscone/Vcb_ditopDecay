text2workspace.py $1 -o workspace.root --channel-masks | tee results/workspace.out
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2  | tee results/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root results/combine.root

