#!/usr/bin/env bash

# Usage: ./extract_cdf.sh ${data_folder} {trace}

data_folder=$1
trace=$2

python ./models/utils/cdf.py --data-folder ${data_folder} --trace ${trace}
