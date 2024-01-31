#!/bin/bash
# Usage ./script_scheduling.sh {results_folder} {data_folder} {trace}

results_folder=$1
data_folder=$2
trace=$3

# Extract hint-based approaches (i.e., DUMBO and pHeavy) performances for simulation
# DUMBO
tpr_dumbo=`cat ${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rates.txt | awk -F "," 'NR==1 {print $2}'`
tnr_dumbo=`cat ${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rates.txt | awk -F "," 'NR==2 {print $2}'`
echo "Launching scheduling simulation with DUMBO TPR: $tpr_dumbo and TNR: $tnr_dumbo"
# pHeavy
tpr_pheavy=`cat ${results_folder}/5_pk/tcp_udp/pheavy_5-20_thr0.6_0dryrun_${trace}/rates.txt | awk -F "," 'NR==1 {print $2}'`
tnr_pheavy=`cat ${results_folder}/5_pk/tcp_udp/pheavy_5-20_thr0.6_0dryrun_${trace}/rates.txt | awk -F "," 'NR==2 {print $2}'`
echo "Launching scheduling simulation with pHeavy TPR: $tpr_pheavy and TNR: $tnr_pheavy"

# Run the scheduling simulator. Warning: needs Python2.7 installed for legacy reasons
cd simulator
automake --add-missing
autoreconf
./configure
make
cd ./py
python2 ./run_experiments_${trace}.py \
--tpr-dumbo $tpr_dumbo --tnr-dumbo $tnr_dumbo \
--tpr-pheavy $tpr_pheavy --tnr-pheavy $tnr_pheavy

# Copy results for plotting
cd ../..
cp ./simulator/py/result_*_${trace}_*.txt $results_folder/simu_output/scheduling/
