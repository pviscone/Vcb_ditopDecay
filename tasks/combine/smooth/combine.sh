

mkdir -p $2
mkdir -p $2/root
rm $2/*.{root,out,json,pdf} $2/root/*
text2workspace.py $1 -o workspace.root | tee $2/workspace.out
combine workspace.root -M MultiDimFit -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2  | tee $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/$2.root
mv workspace.root $2/

combineTool.py -M Impacts -d $2/workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 --robustFit 1 --doFits --parallel 30 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 -o $2/impacts.json -t -1 --expectSignal=1  | tee -a $2/combine_log.out
plotImpacts.py -i $2/impacts.json -o $2/impact_plot | tee -a $2/combine_log.out
mv higgsCombine* $2/root/
mv combine_logger.out $2/
