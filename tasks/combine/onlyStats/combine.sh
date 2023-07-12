#! /bin/bash
text2workspace.py $1 -o workspace.root --channel-masks | tee log/workspace.out
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2 --setParameters mask_Muons=1 | tee log/Electrons_$2.out
mv higgsCombineTest.MultiDimFit.mH120.root results/Electrons_$2.root 


combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2 --setParameters mask_Electrons=1  | tee log/Muons_$2.out
mv higgsCombineTest.MultiDimFit.mH120.root results/Muons_$2.root 

combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2  | tee log/both_$2.out
mv higgsCombineTest.MultiDimFit.mH120.root results/both_$2.root 
