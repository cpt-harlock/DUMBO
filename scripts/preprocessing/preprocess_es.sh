#!/bin/bash

# parameters:
trace_set=${1}  # "caida", "mawi", "uni"
proto=${2}  # "tcp_udp", "tcp"

[ "${proto}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
[ "${proto}" = "tcp" ] && proto_r="tcp" || proto_r="all"

if [[ "${trace_set}" == "caida" ]]; then
    minutes=( 134600 134700 134800 134900 135000 135100 135200 135300 135400 135500 );
elif [[ "${trace_set}" == "mawi" ]]; then
    minutes=( 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 );
elif [[ "${trace_set}" == "uni" ]]; then
    minutes=( 146 147 148 149 151 152 153 154 155 156 );
else
    echo "unknown trace set '${trace_set}'"
    exit 255
fi

function get_trace() {
    local mm=$1
    if [[ "${trace_set}" == "caida" ]]; then
        echo "traces/${trace_set}/20160121-${mm}.UTC.anon.pcap"
    elif [[ "${trace_set}" == "mawi" ]]; then
        echo "traces/${trace_set}/${mm}.pcap"
    elif [[ "${trace_set}" == "uni" ]]; then
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

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    trace_es=$(get_trace_es "${minute}" "${proto_r}")
    ./target/release/elastic_sketch_pcap_parser "$trace" "$trace_es" $proto_flag &
done
wait
