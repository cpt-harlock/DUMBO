#!/usr/bin/env bash

# Usage: ./run.sh {trace}

trace=${1} # valid values: "caida", "mawi", "uni"


#############################
# SETUP
#############################

data_folder="./data/"
results_folder="./results/"
eval "$(conda shell.bash hook)"
conda activate ./conda_envs/training_env
chmod +x ./scripts/*/*.sh
chmod +x ./scripts/*/*.out


#############################
# PCAP TRACES PREPROCESSING
#############################

./scripts/preprocessing/pcap_parsing.sh $data_folder $trace
./scripts/preprocessing/features_extraction.sh $data_folder $trace


#############################
# MODELS TRAINING AND TESTING
#############################

# Models for single protocol
if [[ "$trace" == "uni" ]]; then
    echo "Skipping single-protocol models for UNI trace.";
else
    ./scripts/models/training.sh $data_folder $results_folder $trace "tcp" 1 none 0 0
    ./scripts/models/training.sh $data_folder $results_folder $trace "udp" 1 none 0 0
    ./scripts/models/training.sh $data_folder $results_folder $trace "icmp" 1 none 0 0
fi

# Models for TCP
if [[ "$trace" == "caida" ]]; then
    ./scripts/models/training.sh $data_folder $results_folder $trace "tcp" 5 pruning+feat_selection+quantization 0 0
    ./scripts/models/pheavy.sh $data_folder $results_folder $trace "tcp" 0
fi

# Models for TCP+UDP
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 5 pruning+feat_selection+quantization 0 0
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 1 pruning+feat_selection+quantization 0 0
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 5 ultrapruning+feat_selection+quantization "${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 5 feat_selection+quantization "${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 5 none "${results_folder}/5_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "tcp_udp" 1 none "${results_folder}/1_pk/tcp_udp/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/pheavy.sh $data_folder $results_folder $trace  "tcp_udp" 0

# Models for TCP+UDP+ICMP
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 5 pruning+feat_selection+quantization 0 0
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 1 pruning+feat_selection+quantization 0 0
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 5 ultrapruning+feat_selection+quantization "${results_folder}/5_pk/all_proto/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 5 feat_selection+quantization "${results_folder}/5_pk/all_proto/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 5 none "${results_folder}/5_pk/all_proto/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/training.sh $data_folder $results_folder $trace "all_proto" 1 none "${results_folder}/1_pk/all_proto/initial5min_pruning+feat_selection+quantization_0dryrun_${trace}/rs_cv_results.csv" 0 random
./scripts/models/pheavy.sh $data_folder $results_folder $trace "all_proto" 0


#############################
# USE CASES SIMULATIONS
#############################

# Prepare models for use cases simulations (DUMBO ONNX and pHeavy confusion matrix)
./scripts/models/sklearn2onnx.sh $data_folder $results_folder $trace "tcp_udp" 5 pruning+feat_selection+quantization
./scripts/models/confusion_matrix.sh $results_folder $trace "tcp_udp" 5
if [[ "$trace" == "caida" ]]; then
    ./scripts/models/sklearn2onnx.sh $data_folder $results_folder $trace "tcp" 5 pruning+feat_selection+quantization
    ./scripts/models/confusion_matrix.sh $results_folder $trace "tcp" 5
fi

# Simulations setup. Warning: first, deactivate anaconda with `conda deactivate` (Rust constraint)
conda deactivate
./scripts/preprocessing/init_trace_dir.sh $data_folder $trace
./scripts/preprocessing/preprocess_es.sh $trace "tcp_udp"
if [[ "$trace" == "caida" ]]; then
    ./scripts/preprocessing/preprocess_es.sh $trace "tcp"
fi
conda activate ./conda_envs/training_env
./scripts/preprocessing/preprocess_synth.sh $results_folder $trace "tcp_udp" && wait
if [[ "$trace" == "caida" ]]; then
    ./scripts/preprocessing/preprocess_synth.sh $results_folder $trace "tcp" && wait
fi
# Flow size estimation
conda deactivate
./scripts/use_cases/script_fse.sh $results_folder $trace "tcp_udp" 5 pruning+feat_selection+quantization
if [[ "$trace" == "caida" ]]; then
    ./scripts/use_cases/script_fse.sh $results_folder $trace "tcp" 5 pruning+feat_selection+quantization
fi

# Inter-arrival time
./scripts/use_cases/script_iat.sh $results_folder $trace "tcp_udp" 5 pruning+feat_selection+quantization
if [[ "$trace" == "caida" ]]; then
./scripts/use_cases/script_iat.sh $results_folder $trace "tcp" 5 pruning+feat_selection+quantization
fi

# Scheduling. Warning: first, apply the scheduling_DUMBO.patch on the YAPS repository.
conda activate ./conda_envs/training_env
./scripts/preprocessing/extract_cdf.sh $data_folder $trace # Extract the trace CDF for the simulator
mkdir ./simulator/py/$trace
cp $data_folder/$trace/CDF_$trace.txt ./simulator/py/
conda activate ./conda_envs/scheduling_env
./scripts/use_cases/script_scheduling.sh $results_folder $data_folder $trace


#############################
# MISPREDICTIONS ANALYSIS
#############################

# Synthetically simulate various DUMBO performances. Warning: runs only on the CAIDA trace
if [[ "$trace" == "caida" ]]; then
    echo "Running sensitivity analysis..."
    conda deactivate

    # Flow size estimation
    ./scripts/use_cases/script_fse_synth.sh "caida" "tcp_udp"
    ./scripts/use_cases/script_fse_synth.sh "caida" "tcp"

    # IAT estimation
    ./scripts/use_cases/script_iat_synth.sh "caida" "tcp_udp"
    ./scripts/use_cases/script_iat_synth.sh "caida" "tcp"

    # Scheduling
    conda activate ./conda_envs/training_env
    ./scripts/preprocessing/extract_cdf.sh $data_folder "caida"  # Extract the trace CDF for the simulator
    mkdir ./simulator/py/$trace
    cp $data_folder/$trace/CDF_$trace.txt ./simulator/py/
    mkdir ./simulator/py/synth
    conda activate ./conda_envs/scheduling_env
    ./scripts/use_cases/script_scheduling_synth.sh $results_folder $data_folder
    conda activate ./conda_envs/scheduling_env
    ./scripts/use_cases/script_scheduling_synth.sh $results_folder $data_folder
fi
