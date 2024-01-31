#!/bin/bash

# parameters:
trace_set=${1}  # "caida", "mawi", "uni"
proto=${2}  # "tcp_udp", "tcp"

[ "${proto}" = "tcp" ] && proto_flag="--tcp_only" || proto_flag=""
[ "${proto}" = "tcp" ] && proto_r="tcp" || proto_r="all"
if [[ "${trace_set}" == "caida" ]]; then
    hh_pct=0.02
    trace=traces/caida/20160121-135000.UTC.anon.pcap
    trace_prefix=traces/caida/20160121-135000.UTC.anon
elif [[ "${trace_set}" == "mawi" ]]; then
    hh_pct=0.015
    trace=traces/mawi/1920.pcap
    trace_prefix=traces/mawi/1920
elif [[ "${trace_set}" == "uni" ]]; then
    hh_pct=0.03
    trace=traces/uni/147.pcap
    trace_prefix=traces/uni/147
else
    echo "unknown trace set '${trace_set}'"
    exit 255
fi

thresholds="$trace_prefix"."$proto_r"."$hh_pct"_thresholds.csv
fnr_to_fpr="$trace_prefix"."$proto_r"."$hh_pct"_fnr_to_fpr.csv
#fpr_to_fnr="$trace_prefix"."$proto_r"."$hh_pct"_fpr_to_fnr.csv

for ms in $(seq 500 50 500); do  # iat use case takes 1-2 order of magnitude more memory then the mode, hence effect is negligible

  # oracle
  python ./simulations/run_simulation_iat.py --type oracle --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold 0.5 --ms "$ms" $proto_flag --silent &

  # varying ap score
  for i in $(seq 2 1 "$(wc -l < $thresholds)"); do
    ap=$(awk -v i="$i" -F ',' 'FNR == i {print $1}' $thresholds)
    threshold=$(awk -v i="$i" -F ',' 'FNR == i {print $2}' $thresholds)
    python ./simulations/run_simulation_iat.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --proba_threshold "$threshold" --ap "$ap" --ms "$ms" $proto_flag --silent &
  done

  # varying false negative rate
  for i in $(seq 2 2 "$(wc -l < $fnr_to_fpr)"); do
    fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
    fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
    python ./simulations/run_simulation_iat.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$fnr" --fpr "$fpr" --ms "$ms" $proto_flag --silent &
  done

  wait

  # varying false positive rate
#  for i in $(seq 2 1 "$(wc -l < $fpr_to_fnr)"); do
#    fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
#    fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
#    python ./simulations/run_simulation_iat.py --type synth --pcap "$trace" --hh_perc "$hh_pct" --fnr "$fnr" --fpr "$fpr" --ms "$ms" --silent $proto_flag &
#  done

  wait
  echo "Finished simulating models."

  # oracle - error
  python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=oracle_"${ms}"KB $proto_flag

  for i in $(seq 2 1 "$(wc -l < $thresholds)"); do
    ap=$(awk -v i="$i" -F ',' 'FNR == i {print $1}' $thresholds)
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=sim_ap"$ap"_"$ms"KB $proto_flag &
  done

  for i in $(seq 2 2 "$(wc -l < $fnr_to_fpr)"); do
    fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
    fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fnr_to_fpr)
    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=sim_fnr"$fnr"_fpr"$fpr"_"$ms"KB $proto_flag &
  done

#  for i in $(seq 2 1 "$(wc -l < $fpr_to_fnr)"); do
#    fnr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $2); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
#    fpr=$(awk -v i="$i" -F ',' 'FNR == i {a = sprintf("%.3f", $1); sub(/0+$/, "", a); sub(/\.$/, ".0", a); print a}' $fpr_to_fnr)
#    python ./simulations/error_iat.py --trace "$trace" --perc="$hh_pct" --model=sim_fnr"$fnr"_fpr"$fpr"_"$ms"KB $proto_flag &
#  done

  wait
  echo "Computed error metrics."

done
