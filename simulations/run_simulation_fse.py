import argparse
import os
import subprocess
import math
import sys

from utils import model_string


NEUTRAL_MODEL_SIZE = 531*1024  # a model of this size does not steal any other resource from the downstream tasks
NEUTRAL_FLOW_MANAGER_SIZE = 290*1024  # a flow manager (former packet cache) of this size does not steal any other resource from the downstream tasks
NEUTRAL_ELEPHANT_TRACKER_SIZE = 117*1024  # a flow manager (former packet cache) of this size does not steal any other resource from the downstream tasks
MIN_FM_ROWS = 30


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, required=True, help="oracle | random | synth | baseline | onnx | onnx-pre-bins")
parser.add_argument("--pcap", type=str, required=True)
parser.add_argument("--hh_perc", type=float, required=True)
parser.add_argument("--proba_threshold", type=float, required=False)
parser.add_argument("--ap", type=float, required=False, help="desired AP score for synthetic model (in alternative, use fnr and fpr)")
parser.add_argument("--fnr", type=float, required=False, help="desired rate of mispredicted elephants for synthetic model")
parser.add_argument("--fpr", type=float, required=False, help="desired rate of mispredicted mice for synthetic model")
parser.add_argument("--ms", type=int, required=False, default=531, help="synthetic model size (KB)")
parser.add_argument("--model", type=str, required=False, help="model file (required if type is onnx)")
parser.add_argument("--features", type=str, required=False, help="model features file (required if type is onnx)")
parser.add_argument("--bins", type=str, required=False, help="model features file (required if type is onnx-bins)")
parser.add_argument("--fse_mem", type=float, required=False, default=1.0, help="memory dedicated to the use case (MB) - should not include generic pipeline neutral memory")
parser.add_argument("--tcp_only", required=False, default=False, action='store_true', help="only consider tcp traffic")
parser.add_argument("--silent", required=False, default=False, action='store_true', help="suppress simulator stdout")
parser.add_argument("--bf", required=False, default=False, action='store_true', help="add a BF to remember UDP flows (we assume 1/10 of flows)")
parser.add_argument("--pheavy", required=False, default=False, action='store_true', help="setup the pipeline memory pheavy-like")
args = parser.parse_args()

# specify either ap or both fnr,fpr
if args.type == 'synth' and ((not args.ap and (not args.fnr and not args.fpr)) or (args.ap and (args.fnr or args.fpr))):
    parser.print_help()
    sys.exit(-2)

# threshold is mandatory if model type is synth with ap
if args.type == 'synth' and args.ap:
    if not args.proba_threshold:
        parser.print_help()
        sys.exit(-2)


# model and features file is mandatory when model type is onnx
if args.type == 'onnx':
    if not args.model or not args.features:
        parser.print_help()
        sys.exit(-2)

# model and features file is mandatory when model type is onnx-bins
if args.type == 'onnx-pre-bins':
    if not args.model or not args.features or not args.bins:
        parser.print_help()
        sys.exit(-2)

# flow fingerprint size
flow_fingerprint_size = 6
flow_counter_size = 3
# 13 bytes key +  32-bits counters
hh_bucket_size = (flow_fingerprint_size+flow_counter_size)
#hh_load_factor = (1.0/0.7)
hh_load_factor = 0.95
ht_slots = 8
ht_count = 8
'''
We were used to compute  hh_buckets and flow_manager_capacity based on the number of flows from each trace family.
- old way: # hh_buckets = int(math.floor((avg_flows_per_trace*args.hh_perc)/hh_load_factor))
- old way: # flow_manager_capacity = avg_flows_per_trace*0.019504153779192627
Instead, we now just fix the following values (initially computed using 951182, 601347, and 5000 as flow sizes):
'''
if 'caida' in args.pcap:
    hh_buckets = 20024
    flow_manager_capacity = 18552
elif 'mawi' in args.pcap:
    hh_buckets = 9494
    flow_manager_capacity = 11728
elif 'uni' in args.pcap:
    hh_buckets = 110
    flow_manager_capacity = 100
else:
    hh_buckets = 20000
    flow_manager_capacity = 20000
ht_rows = int(math.floor(hh_buckets/(ht_slots*ht_count)))
# model memory
model_memory = args.ms*1024
# flow manager
# same amount of rows as hh hash table
flow_manager_load_factor = 1.0
flow_manager_slots = 8
flow_manager_slot_size = (flow_fingerprint_size + 8 + 2)
# just to define the variable
flow_manager_rows = 0
# bloom filter accounts for 10% of flows (UDP flows)
p = 0.001
k = 4

# cms
cms_rows = 2
# baseline buckets are all 3 bytes
baseline_cms_bucket_size = 3
# 3 bytes first row + 2 bytes other rows 
cms_bucket_size_first_row = 3 
cms_bucket_size_other_rows = 2
# bf - currently not used (the bf is just simulated by the Flow Manager implementation)
bf_n = round(100000)
bf_m = math.ceil((bf_n * math.log(0.001)) / math.log(1 / 2**math.log(2))) if args.bf else 0
bf_k = round((bf_m/bf_n)*math.log(2))

threshold = args.proba_threshold

process_vector = []

mem = args.fse_mem
trace_set, trace_name = args.pcap.split('/')[-2:]
trace = f"{trace_set}-{trace_name.split('.')[0]}"
output_dir = "./results/simu_output/{}/{}/top_{}_pct/fse/memory_{}MB".format(trace, 'tcp' if args.tcp_only else 'all', args.hh_perc, mem)

output_suffix = model_string(args.type, args.ms, args.ap, args.fnr, args.fpr, args.pheavy)

flow_manager_rows = int(math.floor(flow_manager_capacity/flow_manager_load_factor)/flow_manager_slots)
flow_manager_rows = max(MIN_FM_ROWS, flow_manager_rows)

if args.pheavy:
    #fm_ratio = flow_manager_slot_size/(17 + flow_fingerprint_size)
    fm_ratio = 1
    flow_manager_slot_size = (17 + flow_fingerprint_size)
    if 'uni' not in args.pcap:
        flow_manager_rows = 75000/flow_manager_slots
    else:
        flow_manager_rows = 1500/flow_manager_slots
    flow_manager_rows = int(flow_manager_rows*fm_ratio*0.96)

if 'uni' in args.pcap and not args.pheavy:
    # we use a non hierarchical flow manager on uni as it has a big impact on error when small amount of flows
    flow_manager_rows = int(flow_manager_rows*0.96)
    fm_type = "evict-oldest"
else:
    fm_type = "layered"

flow_manager_mem = flow_manager_slot_size*flow_manager_rows*flow_manager_slots

# memory computations in bytes
remaining_memory = mem*1024*1024 - hh_buckets*flow_counter_size                         # [FSE] elephant tracker
remaining_memory = remaining_memory - (model_memory - NEUTRAL_MODEL_SIZE)               # [pipeline] model
#remaining_memory = remaining_memory - (et_memory - NEUTRAL_ELEPHANT_TRACKER_SIZE)      # [pipeline] elephant tracker
#remaining_memory = remaining_memory - math.ceil(bf_m/8)                                # [pipeline] bf
remaining_memory = remaining_memory - (flow_manager_mem - NEUTRAL_FLOW_MANAGER_SIZE)    # [pipeline] flow manager

if args.type == 'baseline':
    remaining_memory += hh_buckets*flow_counter_size
    remaining_memory += (model_memory - NEUTRAL_MODEL_SIZE)
    remaining_memory += (flow_manager_mem - NEUTRAL_FLOW_MANAGER_SIZE)
    #remaining_memory += (et_memory - NEUTRAL_ELEPHANT_TRACKER_SIZE)

if 'uni' in args.pcap and args.type != 'baseline':
    remaining_memory += (model_memory - NEUTRAL_MODEL_SIZE)
    remaining_memory += (flow_manager_mem - NEUTRAL_FLOW_MANAGER_SIZE)

cms_mem = remaining_memory

if cms_mem < 0:
    print("not enough memory for allocating cms")
    exit(1)

cms_mem_per_column = cms_bucket_size_first_row + (cms_bucket_size_other_rows * (cms_rows - 1))
cms_cols = int(math.floor(cms_mem/cms_mem_per_column))

# FIXME empirically it seems better to use multiples of 16, why? (rust built-in hash function performing strangely)
#if 'caida' in args.pcap:
#    cms_cols = int(cms_cols - math.fmod(cms_cols, 16))
#elif 'uni' in args.pcap:
#    cms_cols = int(cms_cols - math.fmod(cms_cols, 2))

if 'caida' in args.pcap:
    fm_proportions = [1.0, 0.7144182260295988, 0.578189415431055, 0.508657433933787]
elif 'mawi' in args.pcap:
    fm_proportions = [1.0, 0.1494535863287796, 0.11029234944518873, 0.09001225279695857]
elif 'uni' in args.pcap:
    fm_proportions = [1.0, 0.7321054834658853, 0.5963443560764616, 0.5228128924236082]
else:
    fm_proportions = [1.0, 0.85, 0.7, 0.62]

fm_proportions_str = "'{}'".format(" ".join([str(x) for x in fm_proportions]))

# prepare simulation string
cmd_string = ""
ground_truth_file = args.pcap[:-5] + f".{'tcp' if args.tcp_only else 'all'}.gt.csv"
if args.type == "random":
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("random", args.pcap, "-", "-", "-", output_dir, output_suffix, ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, 0.5, 0.0, 0.0, 0.0, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "oracle":
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("oracle", args.pcap, ground_truth_file, "-", "-", output_dir, output_suffix, ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, 0.5, 0.0, 0.0, 0.0, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "synth" and args.pheavy:
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("synth-rates", args.pcap, ground_truth_file, "-", "-", output_dir, output_suffix, ht_rows, ht_slots, ht_count, cms_rows, cms_cols, "evict-oldest", flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, "-", bf_m, bf_k, 0.5, 0.0, args.fnr, args.fpr, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "synth" and args.ap:
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("synth-ap", args.pcap, ground_truth_file, "-", "-", output_dir, output_suffix, ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, threshold, args.ap, 0.0, 0.0, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "synth":
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("synth-rates", args.pcap, ground_truth_file, "-", "-", output_dir, output_suffix, ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, 0.5, 0.0, args.fnr, args.fpr, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "onnx":
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("five-tuple-aggr", args.pcap, args.model, args.features, "-", output_dir, "coda", ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, 0.5, 0.0, 0.0, 0.0, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "onnx-pre-bins":
    cmd_string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format("onnx-pre-bins", args.pcap, args.model, args.features, args.bins, output_dir, "coda", ht_rows, ht_slots, ht_count, cms_rows, cms_cols, fm_type, flow_manager_rows, flow_manager_slots, flow_manager_slot_size, 5, fm_proportions_str, bf_m, bf_k, 0.5, 0.0, 0.0, 0.0, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_fse_fm " + cmd_string
elif args.type == "baseline":
    cmd_string = "{} {} {} {} {} {}".format(args.pcap, output_dir, output_suffix, cms_rows, cms_cols, '--tcp-only' if args.tcp_only else '')
    cmd_string = "./target/release/simulation_baseline " + cmd_string
else:
    print(f"Unknown model type '{args.type}'")
    sys.exit(-1)

print("cmd string:", cmd_string)
p = subprocess.Popen([ cmd_string ], shell=True, stdout=(subprocess.DEVNULL if args.silent else None))
p.wait()
