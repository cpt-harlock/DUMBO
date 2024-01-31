# Simulator

The simulator is written in rust and replays an input pcap trace feeding the monitoring pipeline.

## Main source files

- ``src/simulation_fse_lpc.rs:`` Source file for DUMBO system simulation (flow size estimation use case)
- ``src/simulation_iat_lpc.rs:``Source file for DUMBO system simulation (IAT use case)
- ``src/simulation_baseline.rs:`` Source file for Baseline CMS simulation
- ``src/packet:`` defines Packet and PacketReceiver interface are defined
- ``src/parser:`` PCAP trace parser
- ``src/flow_manager:`` flow manager implementation for aggregated feature retrieval
- ``src/model_wrapper:`` object wrapping various model implementation for inference (ONNX, oracle, simulated metrics, etc.)
- ``src/hash_table:`` implements the Elephant Tracker
- ``src/control_plane:`` dummy control plane that registers keys inserted in the CMS
- ``src/cms:`` Count-Min Sketch for Mice Tracking
- ``src/ddsketch:`` DDSketch used both for Mice and Elephant Tracking
- ``src/bloom_filter:`` a BF implementation, optionally used for mice inference cache

Details on the simulation architecture can be find in [README_DEV.md](README_DEV.md).

## Update binutils

If you run an old version of Ubuntu, you might get some errors while building the blake3 library. Install a newer 
 version of binutils:

```bash
    $ git clone https://github.com/bminor/binutils-gdb
    $ cd binutils-gdb/
    $ sudo apt install libmpc-dev libmpfr-dev
    $ ./configure
    $ make
    $ sudo make install
```

> **_NOTE:_** Actually, just updating ``ar`` is sufficient: ```sudo cp binutils/ar /usr/local/bin/ar``` (in place of
  ```sudo make install```). 

## Running simulation

> **_TODO:_**  Simulator currently requires nightly toolchain. Switch back to stable once the rust stable release
includes pull request [118133](https://github.com/rust-lang/rust/pull/118133) (it will be included in release 1.76.0).

> **_NB:_**  Any Conda environment should be deactivated first (using ```conda deactivate```).

Build the repository with command  ``` cargo build -r ``` . Then in order to launch a simulation:
```bash
./target/release/simulation_fse_lpc [OPTIONS] <MODEL_TYPE> <PCAP_FILE> <MODEL_FILE> <FEATURES_FILE> <OUTPUT_DIR> <OUTPUT_FILE_NAME> <HT_ROWS> <HT_SLOTS> <HT_COUNT> <CMS_ROWS> <CMS_COLS> <FLOW_MANAGER_ROWS> <FLOW_MANAGER_SLOTS> <FLOW_MANAGER_PACKET_LIMIT> <BF_SIZE> <BF_HASH_COUNT> <MODEL_THRESHOLD> <MODEL_MEMORY> <MODEL_AP> <MODEL_EMR> <MODEL_MMR>
```
- MODEL_TYPE: five-tuple-aggr | five-tuple | oracle | random | synth-ap | synth-rates
- PCAP_FILE: network trace file in pcap format
- MODEL_FILE: the file storing the model data (.onnx if running a pretrained model, ground truth if running an oracle or a synth model)
- FEATURES_FILE: pickle file specifying the features actually used by the model (only for .onnx models)
- OUTPUT_DIR: output folder
- OUTPUT_FILE_NAME: all output files will feature this name to distinguish them from other runs
- HT_ROWS: number of rows for the Elephant Tracker
- HT_SLOTS: number of entries per row in the Elephant Tracker (4 -> 95% load factor)
- CMS_ROWS: number of rows for the CMS
- CMS_COLS:  number of cols for the CMS
- FLOW_MANAGER: number of rows for the Flow Manager
- FLOW_MANAGER_SLOTS: number of entries per row in the Flow Manager (implemented as a hash table)
- FLOW_MANAGER_PACKET_LIMIT: firs k packets collected for each flow to extract features from
- BF_SIZE: number of buckets for the Bloom Filter
- BF_HASH_COUNT: number of hash functions used in the Bloom Filter
- MODEL_THRESHOLD: threshold above which a flow is considered as an Elephant (only used when simulating synth models, 
  please provide 0.5 for .onnx models as the threshold is now hardcoded)
- MODEL_AP: the ap-score to simulate (only used if MODEL_TYPE=synth-ap)
- MODEL_EMR: the Elephants misprediction rate (FNR) to simulate (only used if MODEL_TYPE=synth-rates)
- MODEL_MMR: the Mice misprediction rate (FPR) to simulate (only used if MODEL_TYPE=synth-rates)
- [OPTIONAL] --tcp-only: ignore all non-tcp traffic

The simulator generates a series of sub-folders in the output directory:
- gt: ground true sizes for all the flows in the trace
- cms: size estimate of all the flows sent to the control plane
- ht: dump of Elephant Tracker content
- pc: dump of flows found in Flow Manager at the end of the simulation
- pc_evicted: flows evicted from the Flow Manager
  Inside each of these directory, the simulator outputs a file named <OUTPUT_FILE_NAME>.csv (the same folder contains results
  for all past simulations run with different OUTPUT_FILE_NAME parameter).

All files are .csv where row starts with comma-separated flow ID (5 tuple) and end with flow size. To compare true flow
size with estimate, one should import data in a database managing framework, and then it is a matter of join :)


## Python wrappers

You can rely on some python wrappers to run the simulation with certain configurations based on the use case, and also
ensure that all the needed files are correctly preprocessed. Python scripts are located under the [python](./simulations)
folder. Please refer to the following workflow to run them.

### Preprocess

> **_NB:_**  Only needed to run synthetic model simulations (i.e., simulating a model with a certain confusion matrix
or AP score).

Before running a synth model simulation (i.e., a simulation where we run a fake model simulating a certain AP score or
confusion matrix), you need to pre-process the trace to extract round truth and other parameters:

```bash
$ ./scripts/preprocessing/init_trace_dir.sh <DATA_DIR> <TRACE_SET>
$ ./scripts/preprocessing/preprocess_synth.sh <TRACE_SET> <tcp|tcp_udp>
```

Output files are generated under [.trace/](.trace/).

### Run

Wrapper python scripts configure the memory associated to each component and then run the simulation accordingly:

```bash
$ python ./simulations/run_simulation_fse.py --type <MODEL_TYPE> --pcap <PCAP_FILE> --hh_perc <HH_PERCENTAGE> --proba_threshold <THRESHOLD> --ap <AP_SCORE> --fnr <FNR> --fpr <FPR> --ms <MODEL_SIZE>
```

- ``MODEL_TYPE:`` onnx-pre-bins | five-tuple | oracle | random | synth-ap | synth-rates
- ``PCAP_FILE:`` network trace file in pcap format
- ``HH_PERCENTAGE:`` percentage of flows we want to send to the Elephant Tracker
- ``THRESHOLD:`` classification threshold, only use when MODEL_TYPE=synth-ap
- ``AP_SCORE:`` desired ap-score, only use when MODEL_TYPE=synth-ap
- ``FNR:`` desired false negative rate, only use when MODEL_TYPE=synth-rates
- ``FPR:`` desired false positive rate, only use when MODEL_TYPE=synth-rates
- ``MODEL_SIZE:`` model size in KB
- **[OPTIONAL]** ``--tcp_only:`` ignore non-tcp traffic

The simulation generates output files under the ``output/<TRACE>/<PROTOCOL>/top_<HH_PERCENTAGE>_pct/fse/<STAT_FOLDER>``
folders.   particular:

- ``fm:`` flows left in the Flow Manager at the end of the simulation
- ``fm_evicted:`` flows evicted from the flow manager
- ``gt:`` flow size ground truth for all flows
- ``cms:`` flows added to the CMS (Mice Tracker) and their size estimation
- ``ht:`` flows added to the Elephant Tracker and their size
- ``simulation_configuration:`` a dump of the simulation parameters (e.g., pcap file, model file, component sizes, etc.)

In every ``<STAT_FOLDER>``, the output files will be named identically based on the ``<MODEL_TYPE>`` (e.g., ``coda.csv``
for type ``onnx-pre-bins``, which is our solution).

Same applies for the IAT use case, using [./simulations/run_simulation_iat.py](simulations/run_simulation_iat.py) 
 instead (ddsketch stats are given in place of ``cms`` and ``ht``).

### Error

After running the simulation:

```shell
$ python ./simulations/error_fse.py --trace=<PCAP_FILE> --perc=<HH_PERCENTAGE> --memory=<FSE_MEMORY> --packet_limit=<FLOW_MANAGER_PACKET_LIMIT> --model=<OUTPUT_FILE_NAME>
```

- PCAP_FILE: network trace file in pcap format
- HH_PERCENTAGE: percentage of flows we send to the Elephant Tracker
- FSE_MEMORY: base memory dedicated to the use case structures, currently hardcoded to 1.0MB
- FLOW_MANAGER_PACKET_LIMIT: matches the one passed to the main executable, currently always 5 if run through the python wrapper
- OUTPUT_FILE_NAME: this has to match the model suffix used to name output files concerning the simulation (see above)
- [OPTIONAL] --tcp_only: ignore non-tcp traffic

This generates a file with the same name as the previous outputs (``OUTPUT_FILE_NAME``) under the ``error`` folder. For
the FSE use case, it computes the Average Weighted Absolute Estimation error (AWAE), while for IAT
([python/error_fse.py](python_scripts/error_fse.py)), the script computes the Mean Relative Error on the 50th, 75th, 90th, 95th
and 99th quantiles.
