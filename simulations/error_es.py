import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, type=str)
parser.add_argument("--perc", required=True, type=str)
parser.add_argument("--memory", required=True, type=str)
parser.add_argument("--tcp_only", required=False, default=False, action='store_true', help="only consider tcp traffic")
args = parser.parse_args()

trace_set, trace_name = args.trace.split('/')[-2:]
trace = f"{trace_set}-{trace_name.split('.')[0]}"
top_dir = f"./results/simu_output/{trace}/{'tcp' if args.tcp_only else 'all'}/top_{args.perc}_pct/fse/memory_{args.memory}MB"
output_suffix = 'es'

args.es = "".join([ top_dir, "/es/", "elasticsketch" , ".csv" ])

es = {}

df = pd.read_csv(args.es, sep=',', header=None)
error = df.iloc[:, 5].min()

if not os.path.isdir(f"{top_dir}/error"):
    os.makedirs(f"{top_dir}/error")

with open(f"{top_dir}/error/{output_suffix}.txt", "w") as f:
    print(f"Flow sum: 0", file=f)
    print(f"HT discarded: 0", file=f)
    print(f"Total Error: {error}", file=f)
    print(f"Error HT: 0.0", file=f)
    print(f"Error CMS: {error}", file=f)
    print(f"Error PC: 0.0", file=f)
    print(f"Flows only in PC: 0.0", file=f)
    print(f"Flows both in cms/ht: 0", file=f)
    print(f"Flows in ht: 0", file=f)
