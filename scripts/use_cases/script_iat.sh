#!/bin/bash

# Usage: ./script_iat.sh {results_folder} {trace} {protocol} {n_pk} {pruning}

results_folder=$(echo "${1}" | sed 's/\/$//')
trace_set=${2}
protocol=${3}
n_pk=${4}
pruning=${5}

[ "${protocol}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
if [[ "$trace_set" == "caida" ]]; then
    hh_pct=0.02;
    update_minute=134500;
    minutes=( 134600 134700 134800 134900 135000 135100 135200 135300 135400 135500 );
elif [[ "$trace_set" == "mawi" ]]; then
    hh_pct=0.015;
    update_minute=1915;
    minutes=( 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 );
elif [[ "$trace_set" == "uni" ]]; then
    hh_pct=0.03;
    update_minute=145;
    minutes=( 146 147 148 149 151 152 153 154 155 156 );
fi

model="${results_folder}/${n_pk}_pk/${protocol}/initial5min_${pruning}_0dryrun_${trace_set}/cl_pipeline_${update_minute}.onnx"
features="${results_folder}/${n_pk}_pk/${protocol}/initial5min_${pruning}_0dryrun_${trace_set}/cl_pipeline_${update_minute}_selectedfeatures.pkl"
bins="${results_folder}/${n_pk}_pk/${protocol}/initial5min_${pruning}_0dryrun_${trace_set}/cl_pipeline_${update_minute}_bins_edges.pkl"
pheavy_fnr=$(cat "${results_folder}/${n_pk}_pk/${protocol}/pheavy_5-20_thr0.6_0dryrun_${trace_set}/rates.txt" | grep FNR | awk -F ',' '{print $2}')
pheavy_fpr=$(cat "${results_folder}/${n_pk}_pk/${protocol}/pheavy_5-20_thr0.6_0dryrun_${trace_set}/rates.txt" | grep FPR | awk -F ',' '{print $2}')

function get_trace() {
    local mm=$1
    if [[ "$trace_set" == "caida" ]]; then
        echo "traces/${trace_set}/20160121-${mm}.UTC.anon.pcap"
    elif [[ "$trace_set" == "mawi" ]]; then
        echo "traces/${trace_set}/${mm}.pcap"
    elif [[ "$trace_set" == "uni" ]]; then
        echo "traces/${trace_set}/${mm}.pcap"
    fi
}

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")

    # oracle + baselines
    python ./simulations/run_simulation_iat.py --type oracle --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 $proto_flag --silent --run_baselines &

    # pheavy (synthetic)
    python ./simulations/run_simulation_iat.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$pheavy_fnr" --fpr "$pheavy_fpr" --ms 4 --pheavy $proto_flag --silent &

    # coda
    python ./simulations/run_simulation_iat.py --type onnx-pre-bins --pcap "$trace" --model "$model" --features "$features" --bins "$bins" --hh_perc "$hh_pct" $proto_flag --silent &
done
wait
echo "Finished running simulations."

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")

    # baselines - error
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=baseline $proto_flag &

    # oracle - error
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=oracle_531KB $proto_flag &

    # pheavy (synthetic) - error
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=pheavy $proto_flag &

    # coda - error
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=coda $proto_flag &
done
wait
echo "Computed error metrics."
