#!/usr/bin/env bash

# Usage: ./sklearn2onnx.sh {data_folder} {results_folder} {trace} {protocol (tcp, udp, icmp, tcp_udp, all_proto)} {n_pk} {shrinking_level}

data_folder=${1}
results_folder=${2}
trace=${3}
protocol=${4}
n_pk=${5}
shrinking_level=${6}

if [[ $trace == "caida" ]]; then
    minute=135000;
    model_update=134500;
    eleph_tracker_size=20000;
    share_evicted=0.20;
elif [[ $trace == "mawi" ]]; then
    minute=1920;
    model_update=1915;
    eleph_tracker_size=9494;
    share_evicted=0.15;
elif [[ $trace == "uni" ]]; then
    minute=151;
    model_update=145;
    eleph_tracker_size=110;
    share_evicted=0.15;
fi;

model_path=${results_folder}/${n_pk}_pk/${protocol}/initial5min_${shrinking_level}_0dryrun_${trace}/cl_pipeline_${model_update}.pkl

python ./models/utils/sklearn2onnx.py \
--skl-models ${model_path} \
--features-path ./models/training/params/feature_names_5pk.txt \
--features-selection \
--share-evicted ${share_evicted} \
--eleph-tracker-nentries ${eleph_tracker_size} \
--quantization \
--trace ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${minute}_${protocol}.csv \
--n-pk 5