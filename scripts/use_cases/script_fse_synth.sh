#!/bin/bash

# parameters:
trace_set=${1}  # "caida", "mawi", "uni"
proto=${2}  # "tcp_udp", "tcp"

MAX_CHILDREN=$(nproc --all)

[ "${proto}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
[ "${proto}" = "tcp" ] && proto_r="tcp" || proto_r="all"
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

function get_trace_prefix() {
    local mm=$1
    if [[ "$trace_set" == "caida" ]]; then
        echo "traces/${trace_set}/20160121-${mm}.UTC.anon"
    elif [[ "$trace_set" == "mawi" ]]; then
        echo "traces/${trace_set}/${mm}"
    elif [[ "$trace_set" == "uni" ]]; then
        echo "traces/${trace_set}/${mm}"
    fi
}

# baseline
for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    python ./simulations/run_simulation_fse.py --type baseline --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 $proto_flag --silent &
done
wait
echo "Finished running baselines."

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    python ./simulations/error_fse.py --trace="$trace" --perc="$hh_pct" --memory=1.0 --packet_limit=5 --model=baseline $proto_flag &
done
wait
echo "Computed error metrics for baselines."

# models
for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    trace_prefix=$(get_trace_prefix "${minute}")
    thresholds="$trace_prefix"."$proto_r"."$hh_pct"_thresholds.csv
    fnr_to_fpr="$trace_prefix"."$proto_r"."$hh_pct"_fnr_to_fpr.csv
    fpr_to_fnr="$trace_prefix"."$proto_r"."$hh_pct"_fpr_to_fnr.csv

    for ms in $(seq 150 50 1100); do

        # oracle
        python ./simulations/run_simulation_fse.py --type oracle --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 --ms "$ms" $proto_flag --silent &

        while [[ "$trace_set" == "caida" && $(ps -eo ppid | grep -w $$ | wc -w) -ge $((MAX_CHILDREN-($(wc -l <$thresholds))+1)) ]]; do
            wait -n
        done

        # varying ap score
        for i in $(seq 2 1 "$(wc -l < $thresholds)"); do
            ap=$(awk -v i="$i" -F ',' 'FNR == i {print $1}' $thresholds)
            threshold=$(awk -v i="$i" -F ',' 'FNR == i {print $2}' $thresholds)
            python ./simulations/run_simulation_fse.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold "$threshold" --ap "$ap" --ms "$ms" $proto_flag --silent &
        done

        while [[ "$trace_set" == "caida" && $(ps -eo ppid | grep -w $$ | wc -w) -ge $((MAX_CHILDREN-($(wc -l <$fpr_to_fnr)-2)/2)) ]]; do
            wait -n
        done

        # varying false negative rate
        for i in $(seq 4 2 "$(wc -l < $fnr_to_fpr)"); do
            fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
            fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
            python ./simulations/run_simulation_fse.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$fnr" --fpr "$fpr" --ms "$ms" $proto_flag --silent &
        done

        while [[ "$trace_set" == "caida" && $(ps -eo ppid | grep -w $$ | wc -w) -ge $((MAX_CHILDREN-$(wc -l <$fpr_to_fnr)+1)) ]]; do
            wait -n
        done

        # varying false positive rate
        for i in $(seq 2 1 "$(wc -l < $fpr_to_fnr)"); do
            fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
            fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
            if ! awk -F, -v fnr=$(printf '%.3f' $fnr) -v fpr=$(printf '%.3f' $fpr) '{if (fnr == sprintf("%.3f", $1) && fpr == sprintf("%.3f", $2)) {print "The file contains a matching line."}}' $fnr_to_fpr | grep -q "matching"; then
                python ./simulations/run_simulation_fse.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$fnr" --fpr "$fpr" --ms "$ms" $proto_flag --silent &
            fi
        done
    done
    wait
    echo "Finished simulating models for minute ${minute}."
done
wait
echo "Finished simulating models."

for minute in "${minutes[@]}"; do
    trace=$(get_trace "${minute}")
    trace_prefix=$(get_trace_prefix "${minute}")
    thresholds="$trace_prefix"."$proto_r"."$hh_pct"_thresholds.csv
    fnr_to_fpr="$trace_prefix"."$proto_r"."$hh_pct"_fnr_to_fpr.csv
    fpr_to_fnr="$trace_prefix"."$proto_r"."$hh_pct"_fpr_to_fnr.csv
    for ms in $(seq 150 50 1100); do

        # oracle - error
        python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory=1.0 --packet_limit=5 --model=oracle_"${ms}"KB $proto_flag &

        # sim ap - error
        for i in $(seq 2 1 "$(wc -l < $thresholds)"); do
            ap=$(awk -v i="$i" -F ',' 'FNR == i {print $1}' $thresholds)
            python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory=1.0 --packet_limit=5 --model=sim_ap"$ap"_"$ms"KB $proto_flag &
        done

        # sim fnr - error
        for i in $(seq 4 2 "$(wc -l < $fnr_to_fpr)"); do
            fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
            fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
            python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory=1.0 --packet_limit=5 --model=sim_fnr"$fnr"_fpr"$fpr"_"$ms"KB $proto_flag &
        done

        # sim fpr - error
        for i in $(seq 2 1 "$(wc -l < $fpr_to_fnr)"); do
            fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
            fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
            python ./simulations/error_fse.py --trace "$trace" --perc="$hh_pct" --memory=1.0 --packet_limit=5 --model=sim_fnr"$fnr"_fpr"$fpr"_"$ms"KB $proto_flag &
        done
    done
    wait
    echo "Computed error metrics for minute ${minute}."
done

wait
echo "Computed error metrics."
