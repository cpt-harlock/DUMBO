#!/usr/bin/env bash

# Usage: ./pcap_parsing.sh {data_folder} {trace}

data_folder=$1
trace=$2

if [[ "$trace" == "caida" ]]; then
    for i in `seq 130000 100 135900`; do
        echo $i; \
        ./target/release/pcap_parser \
        ${data_folder}/caida/pcap/equinix-chicago.dirA.20160121-$i.UTC.anon.pcap \
        ${data_folder}/caida/oracles_5-20pk_tcpudpicmp/$i/ \
        25;
    done;

elif [[ "$trace" == "mawi" ]]; then
    mkdir ${data_folder}/mawi/pcap/1minepoch/
    for i in 1830 1845 1900 1915 1930 1945; do
        echo "Splitting PCAP into 1-min epochs"
        editcap -i 60 ${data_folder}/mawi/pcap/20190409${i}.pcap ${data_folder}/mawi/pcap/1minepoch/20190409${i} -F pcap;
        for f in ${data_folder}/mawi/pcap/1minepoch/20190409${i}_*_*; do
            fbase=${f%/*};
            fminute=${f#*_}; 
            fminute=${fminute%%_*}; # e.g., 00003
            final=$((${i}))+$((10#$fminute)); 
            final=$(($final));
            echo $final;
            mv "$f" "${fbase}/${final}.pcap"; 
        done;

        end_seq=$((${i}))+15;
        end_seq=$((${end_seq}))
        for j in `seq ${i} 1 ${end_seq}`; do 
            echo $j;
            ./target/release/pcap_parser \
             ${data_folder}/mawi/pcap/1minepoch/${j}.pcap \
             ${data_folder}/mawi/oracles_5-20pk_tcpudpicmp/${j}/ \
            25; 
        done;
    done;

elif [[ "$trace" == "uni" ]]; then
    echo "Merging PCAPs"
    mergecap -F pcap -w ${data_folder}/uni/pcap/univ2.pcap ${data_folder}/uni/pcap/univ2_pt*
    mkdir ${data_folder}/uni/pcap/1minepoch/
    echo "Splitting PCAPs into 1-min epochs"
    editcap -i 60 ${data_folder}/uni/pcap/univ2.pcap ${data_folder}/uni/pcap/1minepoch/univ2 -F pcap
    for f in ${data_folder}/uni/pcap/1minepoch/univ2_*_*; do 
        fbase=${f%/*}; 
        fminute=${f#*_}; 
        fminute=${fminute%%_*}; 
        final=$((10#$fminute)); 
        mv "$f" "${fbase}/${final}.pcap"; 
    done;
    
    for j in `seq 0 1 158`; do
        if [[ ${j} = @(19|29|45|49|64|69|72|87|95|109|117|131|135|150|158) ]]; then
            echo "Removing ${data_folder}/uni/pcap/1minepoch/${j}.pcap (bad split)";
            rm ${data_folder}/uni/pcap/1minepoch/${j}.pcap;
        else
            echo $j;
            ./target/release/pcap_parser \
            ${data_folder}/uni/pcap/1minepoch/${j}.pcap \
            ${data_folder}/uni/oracles_5-20pk_tcpudpicmp/${j}/ \
            25;
        fi;
    done;
fi