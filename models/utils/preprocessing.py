import time
import argparse
import numpy as np
import os
import pandas as pd
import ipaddress
from multiprocess import Pool

from functools import partial
FLAGS = {
    32: "URG",
    16: "ACK",
    8: "PSH",
    4: "RST",
    2: "SYN",
    1: "FIN",
}

def concat_csv(files_list: list, n_pk_parsed: int, dry_run: bool) -> pd.DataFrame:
    dfs = []
    names = [
     'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'min_iat', 'max_iat', 'first_ts', 'last_ts', 'flow_size'
    ] + [
        f"iat_{i}" 
        for i 
        in range(1, n_pk_parsed)
    ] + [
        f"flags_{i}" 
        for i 
        in range(1, n_pk_parsed + 1)
    ] + [
        f"pk_size_{i}" 
        for i 
        in range(1, n_pk_parsed + 1)
    ]
    for f in files_list:
        df = pd.read_csv(f, names=names, header=None)
        dfs.append(df)
    final_df = pd.concat(dfs)

    if dry_run:
        print(f"Dry-run. Using only {dry_run} samples")
        final_df = final_df[:dry_run]
    
    return final_df

def preprocessing_add(row:str) -> list:
    binary_ip = '{:#b}'.format(ipaddress.IPv4Address(row))[2:]
    return list(binary_ip)

def preprocessing_max_sequence(
    df:pd.DataFrame, 
    measure:str="pk_size", 
    sequence_size:int=10,
) -> int:
    # Get list of sequences of measure
    measure_sequence_list = df[[
        measure+ "_" + str(i) 
        for i 
        in range(1, sequence_size)
    ]].values.tolist()
    
    # Apply max for each sequence
    measure_sequence_max = [
        np.nanmax(v) 
        for v 
        in measure_sequence_list
    ]
    
    # Create new feature
    df[measure + "_max_" + str(sequence_size)] = measure_sequence_max
    
    return df

def preprocessing_sum_sequence(
    df:pd.DataFrame, 
    measure:str="pk_size", 
    sequence_size:int=10
) -> int:
    # Get list of sequences of measure
    measure_sequence_list = df[[
        measure+ "_" + str(i) 
        for i 
        in range(1, sequence_size)
    ]].values.tolist()
    
    # Apply sum for each sequence
    measure_sequence_sum = [
        np.nansum(v) 
        for v 
        in measure_sequence_list
    ]
    
    # Create new feature
    df[measure + "_sum_" + str(sequence_size)] = measure_sequence_sum
    
    return df

def preprocessing_min_sequence(
    df:pd.DataFrame, 
    measure:str="pk_size", 
    sequence_size:int=10
) -> int:
    # Get list of sequences of measure
    measure_sequence_list = df[[
        measure+ "_" + str(i) 
        for i 
        in range(1, sequence_size)
    ]].values.tolist()
    
    # Apply min for each sequence
    measure_sequence_min = [
        np.nanmin(v) 
        for v 
        in measure_sequence_list
    ]
    
    # Create new feature
    df[measure + "_min_" + str(sequence_size)] = measure_sequence_min
    
    return df

def count_synack(l: list):
    counter = 0
    for elt in l:
        if not np.isnan(elt):
            for f_number, f_string in FLAGS.items():
                if elt < f_number:
                    continue
                else:
                    elt = elt - f_number
                    if f_string in ["ACK", "SYN"]:
                        counter += 1
    return counter

def count_pshecerst(l: list):
    counter = 0
    for elt in l:
        if not np.isnan(elt):
            for f_number, f_string in FLAGS.items():
                if elt < f_number:
                    continue
                else:
                    elt = elt - f_number
                    if f_string in ["PSH", "ECE", "RST"]:
                        counter += 1
    return counter

def preprocessing_pshecerst_flagcounter(
    df:pd.DataFrame, 
    sequence_size:int,
    protocol:str,
) -> int:
    if protocol == "tcp":
        # Get list of sequences of measure
        measure_sequence_list = df[[
            "flags_" + str(i) 
            for i 
            in range(1, sequence_size)
        ]].values.tolist()
        
        # Apply min for each sequence
        measure_sequence_pshecerst = [
            count_pshecerst(v) 
            for v 
            in measure_sequence_list
        ]
    else:
        measure_sequence_pshecerst = [-1 for elt in range(df.shape[0])]

    # Create new feature
    df["count_pshecerst_" + str(sequence_size)] = measure_sequence_pshecerst
    
    return df

def preprocessing_synack_flagcounter(
    df:pd.DataFrame, 
    sequence_size:int,
    protocol: str,
) -> int:
    if protocol == "tcp":
        # Get list of sequences of measure
        measure_sequence_list = df[[
            "flags_" + str(i) 
            for i 
            in range(1, sequence_size)
        ]].values.tolist()
        
        # Apply min for each sequence
        measure_sequence_synack = [
            count_synack(v) 
            for v 
            in measure_sequence_list
        ]
    else:
        measure_sequence_synack = [-1 for elt in range(df.shape[0])]
        
    # Create new feature
    df["count_synack_" + str(sequence_size)] = measure_sequence_synack
    
    return df

def preprocessing_mean_sequence(
    df:pd.DataFrame, 
    measure:str="pk_size", 
    sequence_size:int=10
) -> int:
    # Get list of sequences of measure
    measure_sequence_list = df[[
        measure + "_" + str(i) 
        for i 
        in range(1, sequence_size)
    ]].values.tolist()
    
    # Apply mean for each sequence
    measure_sequence_mean = [
        np.nanmean(v) 
        for v 
        in measure_sequence_list
    ]
    
    # Create new feature
    df[measure + "_mean_" + str(sequence_size)] = measure_sequence_mean
    
    return df

def preprocessing_std_sequence(
    df:pd.DataFrame, 
    measure:str="pk_size", 
    sequence_size:int=10
) -> int:
    # Get list of sequences of measure
    measure_sequence_list = df[[
        measure+ "_" + str(i) 
        for i 
        in range(1, sequence_size)
    ]].values.tolist()
    
    # Apply std dev for each sequence
    measure_sequence_std = [
        np.nanstd(v) 
        for v 
        in measure_sequence_list
    ]
    
    # Create new feature
    df[measure + "_std_" + str(sequence_size)] = measure_sequence_std
    
    return df

def preprocessing_port(row:int) -> list:
    binary_port = "{0:016b}".format(row)
    return list(binary_port)

def preprocessing_protocol(row:int, protocol: str) -> int:
    if protocol == "tcp":
        res = list("001") 
    elif protocol == "udp":
        res = list("010")
    elif protocol == "icmp":
        res = list("100")
    return res

def preprocess_aggregates(df, pk_start, pk_end, protocol):

    for pk in range(pk_start, pk_end): 
        df = preprocessing_synack_flagcounter(df, pk, protocol) 
        df = preprocessing_pshecerst_flagcounter(df, pk, protocol) 

        df = preprocessing_min_sequence(df, "pk_size", pk+1)
        df = preprocessing_min_sequence(df, "iat", pk)
        
        df = preprocessing_max_sequence(df, "pk_size", pk+1)
        df = preprocessing_max_sequence(df, "iat", pk)

        df = preprocessing_sum_sequence(df, "pk_size", pk+1)
        df = preprocessing_sum_sequence(df, "iat", pk)

        df = preprocessing_mean_sequence(df, "pk_size", pk+1)
        df = preprocessing_mean_sequence(df, "iat", pk)

        df = preprocessing_std_sequence(df, "pk_size", pk+1)
        df = preprocessing_std_sequence(df, "iat", pk)

    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--oracle-files", 
        help="Oracle CSV file(s) to be parsed", 
        type=str, 
        nargs="+", 
        required=True,
    )
    argparser.add_argument(
        "--n-pk-parsed", 
        help="Number of packets parsed in the initial PCAP", 
        type=int, 
        default=25,
        required=True,
    )
    argparser.add_argument(
        "--protocol", 
        help="Protocol used", 
        type=str, 
        choices=["tcp", "udp", "icmp"], 
        required=True,
    )
    argparser.add_argument(
        "--output-file", 
        help="Saving location for preprocessed file", 
        type=str, 
        required=True,
    )
    argparser.add_argument(
        "--verbose", 
        help="Verbosity", 
        action='store_true', 
    )
    argparser.add_argument(
        "--dry-run", 
        type=int,
        help="Uses only n examples", 
        default=0,
    )
    argparser.add_argument(
        "--n-jobs", 
        type=int,
        help="Multiprocessing", 
        default=1,
    )
    argparser.add_argument(
        "--pk-start", 
        type=int,
        help="Minimum n-th packet arrival to compute aggregate features", 
        default=2,
    )
    argparser.add_argument(
        "--pk-end", 
        type=int,
        help="Maximum n-th packet arrival to compute aggregate features", 
        default=22,
    )
    args = argparser.parse_args()

    if args.verbose:
        print("Saving location: " + args.output_file +"\n")
    
    for f in args.oracle_files:
        assert os.path.exists(f), "Input file {0} doesn't exist".format(f)

    df = concat_csv(args.oracle_files, args.n_pk_parsed, args.dry_run)
    if args.verbose:
        print("Parsed CSVs into Pandas DataFrame.")
        print("shape: ", df.shape)
    
    ### Preprocess
    
    # Src and Dst IP adresses
    
    df['src_ip_bin'] = df['src_ip'].map(preprocessing_add)
    df['dst_ip_bin'] = df['dst_ip'].map(preprocessing_add)
    # Break the lists into multiple columns
    df[[
        'src_ip_' + str(i) 
        for i 
        in range(32)
    ]] = pd.DataFrame(
        df['src_ip_bin'].tolist(), 
        index= df.index
    )
    df[[
        'dst_ip_' + str(i) 
        for i 
        in range(32)
    ]] = pd.DataFrame(
        df['dst_ip_bin'].tolist(), 
        index= df.index
    )
    df.drop(
        labels=['src_ip_bin', 'dst_ip_bin'], 
        axis='columns', 
        inplace=True
    )

    # Src and Dst port
    
    df['src_port_bin'] = df['src_port'].map(preprocessing_port)
    df['dst_port_bin'] = df['dst_port'].map(preprocessing_port)
    # Break the lists into multiple columns
    df[[
        'src_port_' + str(i) 
        for i 
        in range(16)
    ]] = pd.DataFrame(
        df['src_port_bin'].tolist(), 
        index=df.index
    )
    df[[
        'dst_port_' + str(i) 
        for i 
        in range(16)
    ]] = pd.DataFrame(
        df['dst_port_bin'].tolist(), 
        index=df.index
    )
    df.drop(
        labels=['src_port_bin', 'dst_port_bin'], 
        axis='columns', 
        inplace=True
    )
    
    # Protocol
    
    df['protocol_bin'] = df['protocol'].apply(lambda row: preprocessing_protocol(row, args.protocol))
    # Break the lists into multiple columns
    df[[
        'protocol_' + i 
        for i 
        in ["icmp", "udp", "tcp"]
    ]] = pd.DataFrame(
        df['protocol_bin'].tolist(), 
        index=df.index
    )
    df.drop(
        labels=['protocol_bin'], 
        axis='columns', 
        inplace=True
    )
    
    # Aggregation for sequences of header values

    df_splits = np.array_split(df, args.n_jobs)
    with Pool(args.n_jobs) as pool:  
        df_list = pool.map(
            partial(
                preprocess_aggregates,
                pk_start=args.pk_start,
                pk_end=args.pk_end,
                protocol=args.protocol,
            ),
            df_splits,
            chunksize=1,
        )
    df = pd.concat(df_list)
    
    # for pk in range(2, 22): 

    #     df = preprocessing_synack_flagcounter(df, pk, args.protocol) 
    #     df = preprocessing_pshecerst_flagcounter(df, pk, args.protocol) 
    #     df = preprocessing_min_sequence(df, "pk_size", pk+1)
    #     df = preprocessing_min_sequence(df, "iat", pk)
    #     df = preprocessing_max_sequence(df, "pk_size", pk+1)
    #     df = preprocessing_max_sequence(df, "iat", pk)
    #     df = preprocessing_sum_sequence(df, "pk_size", pk+1)
    #     df = preprocessing_sum_sequence(df, "iat", pk)
    #     df = preprocessing_mean_sequence(df, "pk_size", pk+1)
    #     df = preprocessing_mean_sequence(df, "iat", pk)
    #     df = preprocessing_std_sequence(df, "pk_size", pk+1)
    #     df = preprocessing_std_sequence(df, "iat", pk)

    ### Save
    df.to_csv(args.output_file, index=False)
    if args.verbose:
        print("Saved pre-processed dataset to " + args.output_file +"\n")