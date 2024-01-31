#!/usr/bin/env bash

# Usage: ./update_CAIDA-MAWI_TCP+UDP.sh {data_folder} {results_folder} {protocols} {n_pk} {shrinking_level} {rs_results} {dry_run_nsamples}

data_folder=${1}
results_folder=${2}
protocols=${3}
n_pk=${4}
shrinking_level=${5}
rs_results=${6}
dry_run_nsamples=${7}

save_folder="${results_folder}/${n_pk}_pk/${protocols}/initial5min_continual250KMbuffer10min_addinit_active+random_${shrinking_level}_updatethrsimu_${dry_run_nsamples}dryrun_driftdetect_caidamawi/"
echo ${save_folder}
mkdir -p ${save_folder}

python ./models/training/training_testing.py \
--simu-drift-detect 57 \
--max-retraining-set-size 2500000 \
--dry-run ${dry_run_nsamples} \
--save-folder ${save_folder} \
--save-models \
--continual \
--shrinking-level ${shrinking_level} \
--w 16 \
--share-evicted 0 \
--init-set add-fifo \
--sampling both \
--sampling-rate 0.01 \
--update-proba-thr-full \
--eleph-tracker-nentries 20000 \
--buffer-size 250000 \
--update-freq 10 \
--min-agreement 0.6 \
--max-model-size 500 \
--n-jobs 50 \
--n-pk ${n_pk} \
--rs-results ${rs_results} \
--hyperparams ./models/training/params/params_pipeline_500KB.json \
--train-init \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130400_${protocols}.csv \
--train-update \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130500_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130600_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130700_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130800_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/130900_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131400_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131500_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131600_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131700_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131800_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/131900_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132400_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132500_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132600_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132700_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132800_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/132900_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133400_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133500_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133600_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133700_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133800_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/133900_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134400_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134500_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134600_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134700_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134800_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/134900_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/135000_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/135100_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/135200_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/135300_${protocols}.csv \
${data_folder}/caida/preprocessed_5-20pk_tcpudpicmp/135400_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1835_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1836_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1837_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1838_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1839_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1840_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1841_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1842_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1843_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1844_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1845_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1846_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1847_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1848_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1849_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1850_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1851_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1852_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1853_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1854_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1855_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1856_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1857_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1858_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1859_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1900_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1901_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1902_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1903_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1904_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1905_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1906_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1907_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1908_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1909_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1910_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1911_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1912_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1913_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1914_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1915_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1916_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1917_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1918_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1919_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1920_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1921_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1922_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1923_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1924_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1925_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1926_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1927_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1928_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1929_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1930_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1931_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1932_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1933_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1934_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1935_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1936_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1937_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1938_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1939_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1940_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1941_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1942_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1943_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1944_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1945_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1946_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1947_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1948_${protocols}.csv \
${data_folder}/mawi/preprocessed_5-20pk_tcpudpicmp/1949_${protocols}.csv \
--features ./models/training/params/feature_names_5pk.txt | tee ${save_folder}/log.txt