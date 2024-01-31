#!/bin/bash

if [ ! $# -eq 2 ]; then
	echo "Usage: ./script_es.sh <TRACE_FILE> <PROTO>"
	exit
fi

echo "${0} ${1} ${2}"

trace_set=$(echo "${1}" | awk -F '/' '{print $(NF-1)}')
trace_name=$(echo "${1}" | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}')
trace="$trace_set-$trace_name"
if [[ $trace_set == *caida* ]]; then
  perc="0.02"
elif [[ $trace_set == *mawi* ]]; then
  perc="0.015"
elif [[ $trace_set == *uni* ]]; then
  perc="0.03"
else
  perc="0.02"
fi

if [[ $trace_set == *uni* ]]; then
  mem_array=(0.01 0.015 0.02)
  ht_mem_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
else
  mem_array=(1.0 1.5 2.0)
  ht_mem_array=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)
fi

mkdir -p elastic_sketch || true
for mem in ${mem_array[@]}; do
  echo "running mem ${mem}..."
	out_dir="./results/simu_output/$trace/${2}/top_${perc}_pct/fse/memory_${mem}MB/es"
	out_file=${out_dir}/elasticsketch.csv
	rm ${out_file} || true
	mkdir -p ${out_dir} || true
	mem_correct=$( bc -l <<< "1000.0*${mem}" )
	mem_correct=$(printf "%.0f" ${mem_correct})
	{ for ht_mem in ${ht_mem_array[@]}; do
		./scripts/use_cases/elastic.out ${1} ${ht_mem} ${mem_correct} 2
	done } | column -t > ${out_file} &
done
wait
