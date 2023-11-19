json_directory="$PWD/json"
echo "$@"
for json in "$json_directory"/*.json; do
	card_path="${json/.json/.txt}"
	card_path="${card_path/json/cards}"
	echo $json
	echo $card_path
	json2datacard $json $card_path "$@"
done

#1: datacard path 2: output path
fit (){
	mkdir -p $2
	text2workspace.py $1 -o workspace.root --channel-masks| tee $2/workspace.out
	#combine workspace.root -M MultiDimFit --algo grid --points 900 -t -1 --expectSignal=1 --saveFitResult --setParameterRanges r=0.8,1.2  $3| tee $2/combine_log.out
	
	#mv higgsCombineTest.MultiDimFit.mH120.root $2/likelihood.root

	#plot1DScan.py $2/likelihood.root -o $2/likelihood_scan --main-label "Stat."  --y-max 3 --y-cut 40| tee $2/plot1DScan.out
	#mv combine_logger.out $2/
	mv workspace.root $2/


	combineTool.py -M Impacts -d $2/workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --expectSignal=1 $3 | tee -a $2/combine_log.out
	combineTool.py -M Impacts -d $2/workspace.root -m 125 --robustFit 1 --doFits --parallel 30 -t -1 --expectSignal=1 $3  | tee -a $2/combine_log.out
	combineTool.py -M Impacts -d $2/workspace.root -m 125 -o $2/impacts.json -t -1 --expectSignal=1 $3 | tee -a $2/combine_log.out
	plotImpacts.py -i $2/impacts.json -o $2/impact_plot | tee -a $2/combine_log.out
	rm higgsCombine*
	mv combine_logger.out $2/
}

#fit "cards/datacard1.txt" "res1"

mkdir -p Muons
mkdir -p Electrons
mkdir -p Combined

for card in "$PWD"/cards/*; do
	res="${card##*/}"
	res="${res/.txt/}"
	fit $card "Muons/$res" "--setParameters mask_Electrons=1"
	fit $card "Electrons/$res" "--setParameters mask_Muons=1"
	fit $card "Combined/$res"
done



