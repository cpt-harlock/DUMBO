#!/usr/bin/env bash

# Usage: ./features_extraction.sh ${data_folder} {trace}

data_folder=$1
trace=$2

mkdir ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/

if [[ $trace == "caida" ]]; then
    seq_start=130000;
    seq_jump=100;
    seq_end=135900;
elif [[ $trace == "mawi" ]]; then
    seq_start=1830;
    seq_jump=1;
    seq_end=1949;
elif [[ $trace == "uni" ]]; then
    seq_start=0;
    seq_jump=1;
    seq_end=158;
fi;

for i in `seq ${seq_start} ${seq_jump} ${seq_end}`; do 

    if [[ "$trace" == "uni" ]] && [[ ${i} = @(19|29|45|49|64|69|72|87|95|109|117|131|135|150|158) ]]; then
        echo "Skipping ${i}";
    elif [[ "$trace" == "mawi" ]] && [[ ${i} = @(1860|1861|1862|1863|1864|1865|1866|1867|1868|1869|1870|1871|1872|1873|1874|1875|1876|1877|1878|1879|1880|1881|1882|1883|1884|1885|1886|1887|1888|1889|1890|1891|1892|1893|1894|1895|1896|1897|1898|1899) ]]; then
        echo "Skipping ${i}";
    else
        echo $i;

        echo "tcp"
        python ./models/utils/preprocessing.py \
        --oracle-files ${data_folder}/${trace}/oracles_5-20pk_tcpudpicmp/${i}/oracle_tcp.csv \
        --protocol tcp \
        --output-file ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_tcp.csv \
        --n-pk-parsed 25 \
        --n-jobs 50 \
        --pk-start 2 \
        --pk-end 22 \
        --verbose;

        echo "udp"
        python ./models/utils/preprocessing.py \
        --oracle-files ${data_folder}/${trace}/oracles_5-20pk_tcpudpicmp/${i}/oracle_udp.csv \
        --protocol udp \
        --output-file ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_udp.csv \
        --n-pk-parsed 25 \
        --n-jobs 50 \
        --pk-start 2 \
        --pk-end 22 \
        --verbose;

        echo "icmp"
        python ./models/utils/preprocessing.py \
        --oracle-files ${data_folder}/${trace}/oracles_5-20pk_tcpudpicmp/${i}/oracle_icmp.csv \
        --protocol icmp \
        --output-file ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_icmp.csv \
        --n-pk-parsed 25 \
        --n-jobs 50 \
        --pk-start 2 \
        --pk-end 22 \
        --verbose;

        awk 'FNR==1 && NR!=1{next;}{print}' \
        ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_tcp.csv \
        ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_udp.csv \
        > ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_tcp_udp.csv;

        awk 'FNR==1 && NR!=1{next;}{print}' \
        ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_tcp.csv \
        ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_udp.csv \
        ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_icmp.csv \
        > ${data_folder}/${trace}/preprocessed_5-20pk_tcpudpicmp/${i}_all_proto.csv;

    fi;
    
done;