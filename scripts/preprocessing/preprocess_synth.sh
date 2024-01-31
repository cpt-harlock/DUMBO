#!/bin/bash

# parameters:
results_dir=${1}  # "./results/"
trace_set=${2}  # "caida", "mawi", "uni"
proto=${3}  # "tcp_udp", "tcp"

[ "${proto}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
if [[ "${trace_set}" == "caida" ]]; then
    hh_pct=0.02
    minutes=( 134600 134700 134800 134900 135000 135100 135200 135300 135400 135500 );
elif [[ "${trace_set}" == "mawi" ]]; then
    hh_pct=0.015
    minutes=( 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 );
elif [[ "${trace_set}" == "uni" ]]; then
    hh_pct=0.03
    minutes=( 146 147 148 149 151 152 153 154 155 156 );
else
    echo "unknown trace set '${trace_set}'"
    exit 255
fi

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
    python ./simulations/preprocess_trace_for_synth_model.py --results_dir="$results_dir" --pcap="$trace" --hh_perc="$hh_pct" $proto_flag &
done
wait
