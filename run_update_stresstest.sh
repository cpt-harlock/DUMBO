#!/usr/bin/env bash

# Usage: ./run_update_stresstest.sh

#############################
# SETUP
#############################

data_folder="./data/"
results_folder="./results/"
eval "$(conda shell.bash hook)"
conda activate ./conda_envs/training_env

#############################
# MODEl UPDATE
#############################

# Model update from CAIDA to MAWI. Warning: requires both traces to have been preprocessed and CAIDA models to have been trained.
./scripts/models/update_CAIDA-MAWI_TCP+UDP.sh $data_folder $results_folder "tcp_udp" 5 pruning+feat_selection+quantization "${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_caida/rs_cv_results.csv" 0
./scripts/models/update_CAIDA-MAWI_TCP+UDP+ICMP.sh $data_folder $results_folder "all_proto" 5 pruning+feat_selection+quantization "${results_folder}/5_pk/all_proto/initial5min_pruning+feat_selection+quantization_0dryrun_caida/rs_cv_results.csv" 0
