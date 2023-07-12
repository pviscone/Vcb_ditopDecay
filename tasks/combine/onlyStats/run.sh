echo "Remember to deactivate the conda enviroment"
./combine.sh datacard_MCstats.txt MCStats
./combine.sh datacard_noMCstats.txt noMCStats


conda activate /eos/user/p/pviscone/newenv
python plot.py
