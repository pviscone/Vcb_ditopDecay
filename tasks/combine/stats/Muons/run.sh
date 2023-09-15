echo "Remember to deactivate the conda enviroment"
rm results/*

./combine.sh datacard_new.txt new_likelihood
./combine.sh datacard_old.txt old_likelihood


