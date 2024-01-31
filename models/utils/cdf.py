import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import matplotlib
from matplotlib.lines import Line2D


PATH_CAIDA = "caida/preprocessed_5-20pk_tcpudpicmp/135000_all_proto.csv"
PATH_MAWI = "mawi/preprocessed_5-20pk_tcpudpicmp/1920_all_proto.csv"
PATH_UNI = "uni/preprocessed_5-20pk_tcpudpicmp/0_all_proto.csv" # UNI trace with the most flows


def main(args):

    # Setup
    data_folder = args.data_folder
    if args.trace == "caida":
        path_trace = PATH_CAIDA
    if args.trace == "mawi":
        path_trace = PATH_MAWI
    if args.trace == "uni":
        path_trace = PATH_UNI
    path = data_folder + path_trace

    all_data = {}
    all_sizes = {}
    all_cutoffs = {}
    all_sizes_cdfs = {}

    # Read trace
    data = pd.read_csv(path, header=0)
    all_data[path] = data
    cutoff = np.percentile(data["flow_size"], 99)
    all_cutoffs[path] = cutoff
    sizes = data["flow_size"]
    all_sizes[path] = sizes

    # Extract CDF
    for path, sizes in all_sizes.items():
        sizes_cdf = pd.DataFrame(data={
                "flow_size": sizes.value_counts().index,
                "n_flows": sizes.value_counts().values,
            })
        sizes_cdf = sizes_cdf.sort_values(by="flow_size", ascending=True)
        sizes_cdf["cumsum"] = sizes_cdf["n_flows"].cumsum()
        sizes_cdf["cdf"] = sizes_cdf["cumsum"]  / sizes_cdf["n_flows"].sum()
        sizes_cdf["tmp"] = 1 # For CDF extraction to .txt file fed to the scheduling simulator
        all_sizes_cdfs[path] = sizes_cdf

    # Save CDF for use by the scheduling simulator
    save_path = f"{data_folder}{args.trace}/CDF_{args.trace}.txt"
    all_sizes_cdfs[path][["flow_size", "tmp", "cdf"]].to_csv(
        save_path, 
        index=False, 
        sep=" "
    )
    print(f"Extracted CDF for {path.split('/')[2]}")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-folder", help="Path to the main data folder", type=str, required=True)
    argparser.add_argument("--trace", help="Trace to consider", choices=["caida", "mawi", "uni"], required=True)
    args = argparser.parse_args()
    
    main(args)