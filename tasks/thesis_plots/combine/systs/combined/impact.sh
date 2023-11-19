text2workspace.py $1 -o workspace.root | tee $2/workspace.out

#full
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  | tee $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood.root
mv workspace.root $2/


combineTool.py -M Impacts -d $2/workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 --robustFit 1 --doFits --parallel 30 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 -o $2/impacts.json -t -1 --expectSignal=1  | tee -a $2/combine_log.out
plotImpacts.py -i $2/impacts.json -o $2/impact_plot | tee -a $2/combine_log.out
rm higgsCombine*
mv combine_logger.out $2/
