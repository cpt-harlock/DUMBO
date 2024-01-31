#!/usr/bin/env bash

# Usage: ./pheavy_confusion_matrix.sh {results_folder} {trace} {protocol (tcp, udp, icmp, tcp_udp, all_proto)} {n_pk} 
results_folder=${1}
trace=${2}
protocol=${3}
n_pk=${4}


# pHeavy
echo "Saving pHeavy confusion matrix"
model_stats_path="${results_folder}/${n_pk}_pk/${protocol}/pheavy_5-20_thr0.6_0dryrun_${trace}/minute_APscore_initial_vs_CL.pkl"
python ./models/utils/confusion_matrix.py \
--model-stats-path ${model_stats_path} \
--pheavy-stage 5

# DUMBO
echo "Saving DUMBO confusion matrix"
model_stats_path="${results_folder}/${n_pk}_pk/${protocol}/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/minute_APscore_initial_vs_CL.pkl"
python ./models/utils/confusion_matrix.py \
--model-stats-path ${model_stats_path} \
--pheavy-stage 0
