#! /bin/bash

text2workspace.py $1 -o workspace_$2.root  | tee log/workspace_$2.out


combine workspace_$2.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2   | tee log/Muons_$2.out
mv higgsCombineTest.MultiDimFit.mH120.root ./results/Muons_$2.root

