#!/bin/bash

# Usage: ./script_fse.sh {results_folder} {trace_set} {protocol} {n_pk} {pruning}

results_folder=$(echo "${1}" | sed 's/\/$//')
trace_set=${2}
protocol=${3}
n_pk=${4}
pruning=${5}

MAX_CHILDREN=$(nproc --all)

[ "${protocol}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
if [[ "$trace_set" == "caida" ]]; then
    hh_pct=0.02;
    update_minute=134500;
    minutes=( 134600 134700 134800 134900 135000 135100 135200 135300 135400 135500 );
    mem_conf=( 1.0 1.5 2.0 );
elif [[ "$trace_set" == "mawi" ]]; then
    hh_pct=0.015;
    update_minute=1915;
    minutes=( 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 );
    mem_conf=( 1.0 1.5 2.0 );
elif [[ "$trace_set" == "uni" ]]; then
    hh_pct=0.03;
    update_minute=145;
    minutes=( 146 147 148 149 151 152 153 154 155 156 );
    mem_conf=( 0.01 0.015 0.02 );
else
    echo "unknown trace set '${trace_set}'"
    exit 255
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

function get_trace_es() {
    local mm=$1
    local pp=$2
    if [[ "${trace_set}" == "caida" ]]; then
            echo traces/caida/20160121-"${mm}".UTC.anon."${pp}".es.dat
        elif [[ "${trace_set}" == "mawi" ]]; then
            echo traces/mawi/"${mm}"."${pp}".es.dat
        elif [[ "${trace_set}" == "uni" ]]; then
            echo traces/uni/"${mm}"."${pp}".es.dat
        fi
}

# baseline
for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    for mem in "${mem_conf[@]}"; do
       python ./simulations/run_simulation_fse.py --type baseline --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 --fse_mem "$mem" --silent $proto_flag &
    done
done
wait

# models
for minute in "${minutes[@]}"; do
    while [[ "$trace_set" == "caida" && $(ps -eo ppid | grep -w $$ | wc -w) -ge $(((MAX_CHILDREN-3)/2)) ]]; do
        wait -n
    done
    trace=$(get_trace "${minute}")
    for mem in "${mem_conf[@]}"; do
        # oracle
        python ./simulations/run_simulation_fse.py --type oracle --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 --fse_mem "$mem" $proto_flag --silent &

        # pheavy (synthetic)
        python ./simulations/run_simulation_fse.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$pheavy_fnr" --fpr "$pheavy_fpr" --ms 4 --fse_mem "$mem" --pheavy $proto_flag --silent &

        # coda
        python ./simulations/run_simulation_fse.py --type onnx-pre-bins --pcap "$trace" --model "$model" --features "$features" --bins "$bins" --hh_perc "$hh_pct" --fse_mem "$mem" $proto_flag --silent &
    done
done

wait
echo "Finished running simulations."

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    for mem in "${mem_conf[@]}"; do
        # baseline - error
        python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory="$mem" --packet_limit=${n_pk} --model=baseline $proto_flag &

        # oracle - error
        python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory="$mem" --packet_limit=${n_pk} --model=oracle_531KB $proto_flag &

        # pheavy (synthetic) - error
        python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory="$mem" --packet_limit=${n_pk} --model=pheavy $proto_flag &

        # coda - error
        python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory="$mem" --packet_limit=${n_pk} --model=coda $proto_flag &
    done
done
wait
echo "Computed error metrics."

# elastic sketch
echo "Running Elastic Sketch baseline..."
[ "${protocol}" = "tcp" ] && proto_r=tcp || proto_r=all
for minute in "${minutes[@]}"; do
    trace_es=$(get_trace_es "${minute}" "${proto_r}")
    env -i "$BASH" -c './scripts/use_cases/script_es.sh '"$trace_es"' '"$proto_r" &
done
wait
for minute in "${minutes[@]}"; do
    trace_es=$(get_trace_es "${minute}" "${proto_r}")
    for mem in "${mem_conf[@]}"; do
        python ./simulations/error_es.py --trace "$trace_es" --perc "$hh_pct" --memory "$mem" $proto_flag
    done
done
echo "Done."
