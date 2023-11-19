mkdir -p $2
382468628480 	382468628480 	323490127872 	430327697408 	482695286784 	181073297408 	316784738304 	799581548544 	675254272
505285152768 	144794869760 	325145186304 	455701217280 	382468628480 	734375256064 	663310495744 	571319435264 	181073297408 	383296139264 	172030173184 	675254272
95625908224 	606173151232 	363155394560 	468930162688 		455701217280 	570778746880 	455701217280 	181073297408 	391028039680 	606173151232 	856804380672
606173151232 	363155394560 	734375256064 	455701217280 		455701217280 		135384297472 	455701217280 	215135498240 	215135498240 	391028039680
606173151232 	606173151232 	269170237440 	468930162688 	455701217280 	455701217280 	599491969024 	135384297472 	910737940480 	141221363712 	606173151232 	391028039680
606173151232 	363155394560 	734375256064 	734375256064 	455701217280 	596144848896 		810267201536 	606173151232 	383296139264 	571319435264 	606173151232
144794869760 	606173151232 	796726849536 	734375256064 	796726849536 	144794869760 	316607541248 	100176330752 	430327697408 	606173151232 	445608792064 	606173151232
144794869760 	606173151232 	795304640512 	

text2workspace.py $1 -o workspace.root | tee $2/workspace.out

#full
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  | tee $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood.root



#freeze btag
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  --freezeNuisanceGroup btag  | tee -a $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood_freeze_btag.root


#freeze btag ctag
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  --freezeNuisanceGroup btag,ctag | tee -a $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood_freeze_bctag.root


#freeze btag ctag jets
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  --freezeNuisanceGroup btag,ctag,JESJER  | tee -a $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood_freeze_bctag_jets.root


#Freeze all nuisanc-es
combine workspace.root -M MultiDimFit --algo grid --points 2000 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.6,1.4  --freezeParameters allConstrainedNuisances | tee -a $2/combine_log.out
mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood_freezed.root

mv workspace.root $2/

cd $2
plot1DScan.py likelihood.root --main-label "Total Uncert." --others likelihood_freeze_btag.root:"freeze btag":4 likelihood_freeze_bctag.root:"freeze btag+ctag":7 likelihood_freeze_bctag_jets.root:"freeze btag+ctag+JESJER":2 likelihood_freezed.root:"stat only":6 --output breakdown --y-max 3 --y-cut 40 --breakdown "btag,ctag,JESJER,norms,stat"
mv breakdown.pdf ../likelihood.pdf
cd ..


# "JESJER,btag,ctag,norms,stat"

combineTool.py -M Impacts -d $2/workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 --robustFit 1 --doFits --parallel 30 -t -1 --expectSignal=1  | tee -a $2/combine_log.out
combineTool.py -M Impacts -d $2/workspace.root -m 125 -o $2/impacts.json -t -1 --expectSignal=1  | tee -a $2/combine_log.out
plotImpacts.py -i $2/impacts.json -o $2/impact_plot | tee -a $2/combine_log.out
rm higgsCombine*
mv combine_logger.out $2/

