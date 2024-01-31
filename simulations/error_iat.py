import numpy as np
import argparse
import os

DISCARD_FLOW_MANAGER_FLOWS_FROM_ERROR = False # If False, we give an advantage to our system because the flows remaining in the PKC at the end of the simu have exact IAT quantiles

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, type=str)
parser.add_argument("--perc", required=True, type=str)
parser.add_argument("--model", type=str, required=True, help="oracle_{ms} | random | sim_ap{ap}_{ms} | sim_fnr{fnr}_fpr{fpr}_{ms} | coda | pheavy | baseline")
parser.add_argument("--tcp_only", required=False, default=False, action='store_true', help="only consider tcp traffic")

args = parser.parse_args()


gt = {}
cms = {}
ht = {}
ht_discarded = set()
pc = {}

trace_set, trace_name = args.trace.split('/')[-2:]
trace = f"{trace_set}-{trace_name.split('.')[0]}"
top_dir = f"./results/simu_output/{trace}/{'tcp' if args.tcp_only else 'all'}/top_{args.perc}_pct/iat"
output_suffix = args.model

quantiles = ["0.5", "0.75", "0.9", "0.95", "0.99"]

def intersection(a,b):
    inter = set()
    for x in a:
        if x in b:
            inter.add(x)
    return inter

def parse_file(filename, dictionary):

    with open(filename, "r") as f:
        for line in f.readlines():
            try:
                rec = line.split(",")
                key = ",".join(rec[:5])
                value = float(rec[5].strip())
                dictionary[key] = value
            except IndexError as e:
                print(filename)
                print(line)
                raise e
    return dictionary

avg_abs_error_model = []
avg_rel_error_model = []
avg_abs_error_one_byte = []
avg_rel_error_one_byte = []
avg_abs_error_two_bytes = []
avg_rel_error_two_bytes = []
avg_abs_error_doubled = []
avg_rel_error_doubled = []

for quantile in quantiles:
    gt_filename = top_dir + f"/gt/{output_suffix}_{quantile}.csv"
    hh_filename = top_dir + f"/hh_ddsketch/{output_suffix}_{quantile}.csv"
    mice_filename = top_dir + f"/mice_ddsketch/{output_suffix}_{quantile}.csv"
    pc_filename = top_dir + f"/flow_manager/{output_suffix}_{quantile}.csv"
    one_byte_per_bucket_ddsketch_filename = top_dir + f"/one_byte_per_bucket_ddsketch/baseline_{quantile}.csv"
    two_bytes_per_bucket_ddsketch_filename = top_dir + f"/two_bytes_per_bucket_ddsketch/baseline_{quantile}.csv"
    doubled_ddsketch_filename = top_dir + f"/doubled_ddsketch/baseline_{quantile}.csv"

    gt = parse_file(gt_filename, {})
    hh = parse_file(hh_filename, {}) if output_suffix != 'baseline' else ''
    mice = parse_file(mice_filename, {}) if output_suffix != 'baseline' else ''
    pc = parse_file(pc_filename, {}) if output_suffix != 'baseline' else ''
    one_byte = parse_file(one_byte_per_bucket_ddsketch_filename, {}) if output_suffix == 'baseline' else ''
    two_bytes = parse_file(two_bytes_per_bucket_ddsketch_filename, {}) if output_suffix == 'baseline' else ''
    doubled = parse_file(doubled_ddsketch_filename, {}) if output_suffix == 'baseline' else ''

    abs_error_model = {}
    rel_error_model = {}
    abs_error_one_byte = {}
    rel_error_one_byte = {}
    abs_error_two_bytes = {}
    rel_error_two_bytes = {}
    abs_error_doubled = {}
    rel_error_doubled = {}

    # Compute error

    for key in gt:
        value = gt[key]
        if output_suffix != 'baseline':
            est_value = 0.0
            if key in hh:
                est_value = hh[key]
            elif key in mice:
                est_value =  mice[key]
            elif key in pc:
                if DISCARD_FLOW_MANAGER_FLOWS_FROM_ERROR:
                    continue
                est_value = value
            else:
                exit(-1)
            abs_err = abs(value - est_value)
            if value != 0.0:
                rel_err = abs_err/value
            else:
                if est_value != 0.0:
                    rel_err = float('inf')
                else:
                    rel_err = 0.0

            abs_error_model[key] = abs_err
            rel_error_model[key] = rel_err
        else:
            # one byte
            one_byte_est_value = one_byte[key]
            abs_error_one_byte[key] = abs(value - one_byte_est_value)
            if value != 0.0:
                rel_error_one_byte[key] = abs(value - one_byte_est_value)/value
            else:
                if one_byte_est_value == 0.0:
                    rel_error_one_byte[key] = 0.0
                else:
                    rel_error_one_byte[key] = float('inf')

            # two byte
            two_bytes_est_value = two_bytes[key]
            abs_error_two_bytes[key] = abs(value - two_bytes_est_value)
            if value != 0.0:
                rel_error_two_bytes[key] = abs(value - two_bytes_est_value)/value
            else:
                if two_bytes_est_value == 0.0:
                    rel_error_two_bytes[key] = 0.0
                else:
                    rel_error_two_bytes[key] = float('inf')

            # doubled
            doubled_est_value = doubled[key]
            abs_error_doubled[key] = abs(value - doubled_est_value)
            if value != 0.0:
                rel_error_doubled[key] = abs(value - doubled_est_value)/value
            else:
                if doubled_est_value == 0.0:
                    rel_error_doubled[key] = 0.0
                else:
                    rel_error_doubled[key] = float('inf')

    if output_suffix != 'baseline':
        avg_abs_error_model.append(np.array(list(abs_error_model.values())).mean())
        avg_rel_error_model.append(np.array(list(rel_error_model.values())).mean())
    else:
        avg_abs_error_one_byte.append(np.array(list(abs_error_one_byte.values())).mean())
        avg_rel_error_one_byte.append(np.array(list(rel_error_one_byte.values())).mean())
        avg_abs_error_two_bytes.append(np.array(list(abs_error_two_bytes.values())).mean())
        avg_rel_error_two_bytes.append(np.array(list(rel_error_two_bytes.values())).mean())
        avg_abs_error_doubled.append(np.array(list(abs_error_doubled.values())).mean())
        avg_rel_error_doubled.append(np.array(list(rel_error_doubled.values())).mean())

try:
    os.makedirs(f"{top_dir}/error")
except FileExistsError:
    pass

with open(f"{top_dir}/error/{output_suffix}.txt", "w") as f:
    for i, quantile in enumerate(quantiles):
        if output_suffix != 'baseline':
            print(f"MRE {quantile} Model:           {avg_rel_error_model[i]}", file=f)
        else:
            print(f"MRE {quantile} 32 bins 1 byte:  {avg_rel_error_one_byte[i]}", file=f)
            print(f"MRE {quantile} 16 bins 2 bytes: {avg_rel_error_two_bytes[i]}", file=f)
            print(f"MRE {quantile} 32 bins 2 bytes: {avg_rel_error_doubled[i]}", file=f)
