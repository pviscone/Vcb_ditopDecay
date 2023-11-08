mkdir -p $2
text2workspace.py $1 -o workspace.root | tee $2/workspace.out
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  | tee $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood.root
mv workspace.root $2/
