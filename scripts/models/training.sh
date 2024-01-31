#!/usr/bin/env bash

# Usage: ./training.sh {data_folder} {results_folder} {trace} {protocol (tcp, udp, icmp, tcp_udp, all_proto)} {n_pk} {shrinking_level} {rs_results} {dry_run_nsamples} {sampling (optional)}

data_folder=${1}
results_folder=${2}
trace=${3}
protocol=${4}
n_pk=${5}
shrinking_level=${6}
rs_results=${7}
dry_run_nsamples=${8}
sampling=${9-both}

save_folder="${results_folder}/${n_pk}_pk/${protocol}/initial5min_${shrinking_level}_${dry_run_nsamples}dryrun_${trace}/"
echo ${save_folder}
mkdir -p ${save_folder}

if [[ $trace == "caida" ]]; then
    seq_start=130000;
    seq_jump=100;
    seq_end=135900;
    ntrain_minutes=5;
    eleph_tracker_size=20000;
    share_evicted=0.20
elif [[ $trace == "mawi" ]]; then
    seq_start=1830;
    seq_jump=1;
    seq_end=1929;
    ntrain_minutes=5;
    eleph_tracker_size=9494;
    share_evicted=0.15
elif [[ $trace == "uni" ]]; then
    seq_start=0;
    seq_jump=1;
    seq_end=158;
    ntrain_minutes=92;
    eleph_tracker_size=110;
    share_evicted=0.15
fi;
traces_list=()
for i in `seq ${seq_start} ${seq_jump} ${seq_end}`; do 
    if [[ "$trace" == "uni" ]] && [[ ${i} = @(19|29|45|49|64|69|72|87|95|109|117|131|135|150|158) ]]; then
        echo "Skipping ${i}";
    elif [[ "$trace" == "uni5min" ]] && [[ ${i} = @(2047|2062|2067|2072|2077|2082|2087|2092|2097|2102|2112|2147|2157|2162|2167|2172|2177|2182|2187|2192|2197|2217) ]]; then
        echo "Skipping ${i}";
    elif [[ "$trace" == "mawi" ]] && [[ ${i} = @(1860|1861|1862|1863|1864|1865|1866|1867|1868|1869|1870|1871|1872|1873|1874|1875|1876|1877|1878|1879|1880|1881|1882|1883|1884|1885|1886|1887|1888|1889|1890|1891|1892|1893|1894|1895|1896|1897|1898|1899) ]]; then
        echo "Skipping ${i}";
    else
        traces_list+=( ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_${protocol}.csv );
    fi;
done;

python ./models/training/training_testing.py \
--dry-run ${dry_run_nsamples} \
--shrinking-level ${shrinking_level} \
--save-folder ${save_folder} \
--update-proba-thr-full \
--eleph-tracker-nentries ${eleph_tracker_size} \
--share-evicted ${share_evicted} \
--continual \
--update-freq 10 \
--min-agreement 0.6 \
--buffer-size 250000 \
--init-set add \
--sampling ${sampling} \
--sampling-rate 0.01 \
--save-models \
--n-splits 2 \
--n-iters 50 \
--max-model-size 500 \
--w 16 \
--n-b-feat-idx 96 97 98 99 \
--n-jobs 50 \
--n-pk ${n_pk} \
--rs-results ${rs_results} \
--hyperparams ./models/training/params/params_pipeline_500KB.json \
--train-init ${traces_list[@]:0:$ntrain_minutes} \
--train-update ${traces_list[@]:$ntrain_minutes:${#traces_list[@]}} \
--features ./models/training/params/feature_names_5pk.txt | tee ${save_folder}/log.txt