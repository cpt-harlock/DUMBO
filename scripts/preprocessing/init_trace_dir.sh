#!/bin/bash

# parameters:
data_folder=$(readlink -f "${1}") # traces location (as used by the model code, e.g., "./data/")
trace_set=${2}   # "caida", "mawi", "uni"

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

function get_trace_orig() {
    local mm=$1
    if [[ "${trace_set}" == "caida" ]]; then
        echo "${data_folder}/${trace_set}/pcap/equinix-chicago.dirA.20160121-${mm}.UTC.anon.pcap"
    elif [[ "${trace_set}" == "mawi" ]]; then
        echo "${data_folder}/${trace_set}/pcap/1minepoch/${mm}.pcap"
    elif [[ "${trace_set}" == "uni" ]]; then
        echo "${data_folder}/${trace_set}/pcap/1minepoch/${mm}.pcap"
    fi
}

function get_trace_ln() {
    local mm=$1
    if [[ "$trace_set" == "caida" ]]; then
        echo "traces/${trace_set}/20160121-${mm}.UTC.anon.pcap"
    elif [[ "$trace_set" == "mawi" ]]; then
        echo "traces/${trace_set}/${mm}.pcap"
    elif [[ "$trace_set" == "uni" ]]; then
        echo "traces/${trace_set}/${mm}.pcap"
    fi
}

mkdir -p ./traces/"$trace_set"
for minute in "${minutes[@]}"; do
    pcap_file=$(get_trace_orig "${minute}")
    link=$(get_trace_ln "${minute}")
    ln -s "$pcap_file" "$link"
done
