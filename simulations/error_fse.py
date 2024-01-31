import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, type=str)
parser.add_argument("--perc", required=True, type=str)
parser.add_argument("--memory", required=True, type=str)
parser.add_argument("--packet_limit", type=int, required=True)
parser.add_argument("--model", type=str, required=True, help="oracle_{ms} | random | sim_ap{ap}_{ms} | sim_fnr{fnr}_fpr{fpr}_{ms} | coda | baseline")
parser.add_argument("--remove_nonexisting_error", default=False, action="store_true")
parser.add_argument("--remove_ht_discarded", default=False, action="store_true")
parser.add_argument("--remove_ht_cms_intersection", default=False, action="store_true")
parser.add_argument("--tcp_only", required=False, default=False, action='store_true', help="only consider tcp traffic")
args = parser.parse_args()

trace_set, trace_name = args.trace.split('/')[-2:]
trace = f"{trace_set}-{trace_name.split('.')[0]}"
top_dir = f"./results/simu_output/{trace}/{'tcp' if args.tcp_only else 'all'}/top_{args.perc}_pct/fse/memory_{args.memory}MB"
output_suffix = args.model

args.gt = "".join([ top_dir, "/gt/", "baseline" , ".csv" ])
args.cms = "".join([ top_dir, "/cms/", output_suffix , ".csv" ])
args.ht = "".join([ top_dir, "/ht/", output_suffix , ".csv" ])
args.pc = "".join([ top_dir, "/fm/", output_suffix , ".csv" ])
args.ht_discarded  = "".join([ top_dir, "/ht_discarded/", output_suffix , ".csv" ])
args.model_predictions = "".join([ top_dir, "/model_predictions/", output_suffix , ".csv" ])

gt = {}
cms = {}
ht = {}
ht_discarded = set()
pc = {}
model_predictions = {}

def compute_sector(k):
    if k in ht:
        return "ht"
    elif k in cms:
        return "cms"
    elif k in pc:
        return "pc"
    return "disc"

with open(args.gt, "r") as f:
    lines = f.readlines()
    for line in lines:
        rec = line.split(",")
        key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4]])
        val = int(rec[5])
        gt[key] = val
with open(args.cms, "r") as f:
    lines = f.readlines()
    for line in lines:
        try:
            rec = line.split(",")
            key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4]])
            val = int(rec[5])
            cms[key] = val
        except (KeyError, IndexError) as err:
            print(f"{top_dir}/error/{output_suffix}.txt")
            print(line)
            raise err
if args.model != 'baseline':
#     with open(args.model_predictions, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             rec = line.split(",")
#             key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4]])
#             val = int(rec[5].replace('[', '').replace(']','').split(',')[0].split('-')[1])
#             print(val)
#             model_predictions[key] = val
    with open(args.ht, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                rec = line.split(",")
                key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4]])
                val = int(rec[5])
                ht[key] = val
            except (KeyError, IndexError) as err:
                print(f"{top_dir}/error/{output_suffix}.txt")
                print(line)
                raise err
    if args.pc and args.packet_limit > 1:
        with open(args.pc, "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    rec = line.split(",")
                    key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4]])
                    val = int(rec[5])
                    pc[key] = val
                except (KeyError, IndexError, ValueError) as err:
                    print(f"{top_dir}/error/{output_suffix}.txt")
                    print(line)
                    raise err
    if args.remove_ht_discarded:
        with open(args.ht_discarded, "r") as f:
            lines = f.readlines()
            for line in lines:
                rec = line.split(",")
                key = ",".join([rec[0], rec[1], rec[2], rec[3], rec[4].strip()])
                ht_discarded.add(key)



#print(len(ht_discarded))

# threshold = np.quantile(list(gt.values()), 0.99)

flow_sum = 0.0
are = 0.0
aae = 0.0
err = 0.0
err_ht = 0.0
err_cms = 0.0
err_pc = 0.0
flows_only_in_pc = 0
fn = 0
fp = 0
tn = 0
tp = 0

intersection_ht_cms = 0

for k in ht:
    if k in cms:
        intersection_ht_cms += 1

error_list = []
estimates = {}

for k in gt:

    v = gt[k]
    v_est = 0.0

    # mispredictions
#     if k in model_predictions:
#         if model_predictions[k] == 0 and v >= threshold:
#             fn += 1
#         elif model_predictions[k] == 0 and v < threshold:
#             tn += 1
#         elif model_predictions[k] == 1 and v < threshold:
#             fp += 1
#         elif model_predictions[k] == 1 and v >= threshold:
#             tp += 1

    # error
    if k in ht_discarded:
        v_est = v
        #continue
    elif args.remove_ht_cms_intersection and k in cms and k in ht:
        continue
    elif k in ht:
        v_est += ht[k]
        if k in cms:
            v_est += cms[k]
        err_ht += abs(v-v_est)*v
    elif k in cms:
        v_est += cms[k]
        if k in pc:
            if pc[k] < args.packet_limit:
                v_est += pc[k]
        err_cms += abs(v-v_est)*v
    elif k in pc:
        v_est += pc[k]
        err_pc += abs(v-v_est)*v
        flows_only_in_pc += 1
    elif args.remove_nonexisting_error:
        continue
    err += abs(v - v_est)*v
    aae += abs(v - v_est)
    are += abs(v - v_est)/v
    estimates[k] = v_est
    flow_sum += v
    error_list.append((k, abs(v - v_est)*v))

size = 400

if not os.path.isdir(f"{top_dir}/error"):
    os.makedirs(f"{top_dir}/error", exist_ok=True)

with open(f"{top_dir}/error/{output_suffix}.txt", "w") as f:
    try:
        for k in ht:
            if ht[k] == gt[k] and k in cms:
                print(f"Flow shouldn't be in cms: {k}", file=f)
        print(f"Flow sum: {flow_sum}", file=f)
        print(f"HT discarded: {len(ht_discarded)}", file=f)
        print(f"Total Error: {err/flow_sum}", file=f)
        print(f"AAE: {aae/len(gt)}", file=f)
        print(f"ARE: {are/len(gt)}", file=f)
        print(f"Error HT: {err_ht/flow_sum}", file=f)
        print(f"Error CMS: {err_cms/flow_sum}", file=f)
        print(f"Error PC: {err_pc/flow_sum}", file=f)
        print(f"Flows only in PC: {flows_only_in_pc}", file=f)
        print(f"Flows both in cms/ht: {intersection_ht_cms}", file=f)
        print(f"Flows in ht: {len(ht)}", file=f)
    except KeyError as err:
        print(err)
        print(f"{top_dir}/error/{output_suffix}.txt")
#     if args.model != 'baseline':
#         print(f"FNR: {fn/(tp+fn)}", file=f)
#         print(f"FPR: {fp/(tn+fp)}", file=f)

#error_list = sorted(error_list, key=lambda a: a[1], reverse=True)
#error_list = [ (x[0],x[1]/flow_sum, compute_sector(x[0]), gt[x[0]], estimates[x[0]]) for x in error_list]
#print("First {} errors: ".format(size), error_list[0:size])
#for i in range(size):
#    print(error_list[i][0], error_list[i][1], error_list[i][2], error_list[i][3], error_list[i][4])
#print("Sum of first {} errors: ".format(size), sum([x[1] for x in error_list[0:size]]))
