#!/bin/bash
# Usage ./script_scheduling_synth.sh {results_folder} {data_folder}

# Copy the synthetic performances of DUMBO on CAIDA (from the Flow Size synthetic simulations)
cp ./traces/caida/20160121-135000.UTC.anon.all.0.02_fnr_to_fpr.csv ./simulator/py/CAIDA_135000.all.0.02_fnr_to_fpr.csv

# Run the scheduling simulator. Warning: needs Python2.7 installed for legacy reasons
cd simulator
automake --add-missing
autoreconf
./configure
make
cd ./py
rm results_synth_caida_*.txt
python2 ./run_experiments_caida_synth.py

# Copy results for plotting
cd ../..
mkdir $results_folder/simu_output/scheduling/synth/
cp ./simulator/py/synth/result_synth_caida_*.txt $results_folder/simu_output/scheduling/synth/
