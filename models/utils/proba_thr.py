import os
import sys
module_path = os.path.abspath(os.path.join('./training/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import voting_rf_classifier
from training_testing import get_threshold

import argparse
import numpy as np
import random
import pandas as pd
import pickle
from copy import deepcopy

import sklearn
from sklearn.ensemble import RandomForestClassifier


SEED = 42
random.seed(SEED)
np.random.seed(SEED)   

            
def main(args):
    print('The scikit-learn version in use is {}.'.format(sklearn.__version__))

    # Load Scikit-learn model
    with open(args.skl_model, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Scikit-learn version used to train the model: {pipeline.__getstate__()['_sklearn_version']}")
    assert pipeline.__getstate__()['_sklearn_version'] == sklearn.__version__
                
    # Load feature names
    with open(args.features_path) as f:
        feature_names = [line.rstrip() for line in f]

    # Compute probability threshold
    proba_thresh = get_threshold(
        model=pipeline,
        n_packets=args.n_pk,
        pct_hh=[args.share_predicted_hh],
        minute_paths=args.traces,
        share_evicted=args.share_evicted,
        features=feature_names,
        dry_run=0,
        eleph_tracker_nentries=[] if args.eleph_tracker_nentries == 0 else [args.eleph_tracker_nentries],
    )[args.eleph_tracker_nentries if args.eleph_tracker_nentries > 0 else args.share_predicted_hh]
    print(f"Proba thresh: {proba_thresh}.")


if __name__ == "__main__":
    
    # example: 
    # --skl-model /mnt/storage/raphael/coda/train_val_models/5_pk/first_pk/TCP+UDP/initial5min_pruning+feat_selection+quantization_0dryrun_caida1/initial_pipeline.pkl
    # --n-pk 5
    # --eleph-tracker-nentries 20000  
    # --share-evicted 0.33765629103758943
    # --features-path ./training/params/feature_names_5pk.txt
    # --traces /mnt/storage/raphael/coda/caida/20160121/preprocessed_5-20pk_tcpudpicmp/135000_tcp.csv 
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--skl-model", help="Path the sklearn pickled model", type=str, required=True)
    argparser.add_argument("--n-pk", help="Model used for the k-th packet arrival", type=int, required=True)
    argparser.add_argument("--eleph-tracker-nentries", help="Number of entries that can be accomodated in the Elephant Tracker", type=int, default=0)
    argparser.add_argument("--share-predicted-hh", help="Share of all flows that should be predicted as Elephant. This arg is ignored if args.eleph_tracker_nentries is not null", type=float, default=0)
    argparser.add_argument("--share-evicted", help="Share of flows > k pk that get evicted from the Flow Manager", type=float, default=0.29)
    argparser.add_argument("--features-path", help="Path to a txt file containing the list of features used as input by the model", type=str, required=True)
    argparser.add_argument("--traces", nargs="+", help="Traces used to infer the probability thr", type=str, required=True)
    
    args = argparser.parse_args()
    
    main(args)