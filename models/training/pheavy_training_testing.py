import os
import sys
module_path = os.path.abspath(os.path.join('./models/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.helpers import get_X_y_binary
from utils.helpers import get_X_y_binary_size_cutoffs
from utils.model_sizing import size, get_tree, get_tree_estimator, get_subforest_estimators
from utils.model_sizing import get_best_trees_idx
from training.voting_rf_classifier import VotingRandomForestClassifier
from utils.meta_cost_learning import MetaCost

import pickle
import numpy as np
import pandas as pd
import random
from scipy.stats import uniform # used for random search space
from statistics import fmean

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import OneSidedSelection

from copy import deepcopy
import random
import time
import argparse
import json
from copy import deepcopy
from pprint import pprint
from multiprocess import Pool
from functools import partial
from types import SimpleNamespace

SEED = 42
random.seed(SEED)
np.random.seed(SEED)   

# We estimate the cost matrix based on heuristics on flow size distribution 
# (i.e. imbalance classification depending on how many flows we excluded < k)
# Shape: actual_class in columns and predicted_class in rows
COSTS = {
    # Shape: actual_class in columns and predicted_class in rows
    5: np.array([[0, 2.5], [1, 0]]),
    6: np.array([[0, 50], [1, 0]]),
    7: np.array([[0, 40], [1, 0]]),
    8: np.array([[0, 37], [1, 0]]),
    9: np.array([[0, 33], [1, 0]]),
    10: np.array([[0, 32], [1, 0]]),
    11: np.array([[0, 29], [1, 0]]),
    12: np.array([[0, 26], [1, 0]]),
    13: np.array([[0, 24], [1, 0]]),
    14: np.array([[0, 22], [1, 0]]),
    15: np.array([[0, 20], [1, 0]]),
    16: np.array([[0, 18], [1, 0]]),
    17: np.array([[0, 17], [1, 0]]),
    18: np.array([[0, 16], [1, 0]]),
    19: np.array([[0, 15], [1, 0]]),
    20: np.array([[0, 14], [1, 0]]),
}



def get_data(train, features, n_pk, dry_run=0, percentile=99):
    """Get the input dataframe and labels"""

    X_preproc, y = get_X_y_binary(
        filenames=train, 
        feature_names=features, 
        min_packets=n_pk,
        percentile=percentile,
        verbose=True,
        dry_run=dry_run,
    )
    X_df = pd.DataFrame(
        data=X_preproc,
        columns=features
    )
    
    return X_df, y

def parse_traces_pheavy(traces, features, n_pk_min, n_pk_max, dry_run, percentile=99):
    data = {}
    print(f"Elephants defined with the {percentile}-th percentile")
    for n in range(n_pk_min, n_pk_max + 1):
        X, y = get_data(traces, features, n, dry_run, percentile)
        if dry_run:
            print(f"Dry-run: capping sizes to {dry_run} examples")
            X = X[:dry_run]
            y = y[:dry_run]
        data[n] = (X, y)

    return data 

def size_pheavy(trees_list, n_b_feat_idx, w):

    sizes = []

    for t in trees_list:
        tree = deepcopy(t)

        # A bit hacky, because the sizing function expects a random forest
        tree_compat_sizing_func = {
            "estimators_": [tree],  
            "tree_": tree.tree_,
        }
        tree_compat_sizing_func = SimpleNamespace(**tree_compat_sizing_func)
        trees_list_compat_sizing_func = {
            "estimators_": trees_list,  
        }
        trees_list_compat_sizing_func = SimpleNamespace(**trees_list_compat_sizing_func)

        sizes_impl = size(
            tree_compat_sizing_func,  
            f_b=tree.n_features_in_ - len(n_b_feat_idx), 
            n_b_feat_idx=n_b_feat_idx,
            w=w, # w =0 -> only binary features
            full_forest=trees_list_compat_sizing_func, # Required for hybrid (N is computed among all trees)
        )
        tree_size = min(sizes_impl.values())
        tree_size = np.ceil(tree_size / 8) / 1_000 # in KB
        impl = min(sizes_impl, key=sizes_impl.get)

        sizes.append({
            "size": tree_size,
            "imp": impl,
        })
    
    return sizes

def score_greedysearch(dt, X_test, y_test, max_depth, size=False):
        y_pred = dt.predict(X_test.values)
        conf_mat = confusion_matrix(y_test, y_pred)
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        tp = conf_mat[1][1]
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        if size:
            tree_size = dt.tree_.node_count
            max_tree_size = 2**(max_depth + 1) - 1
            tree_sparsity = (max_tree_size - tree_size) / max_tree_size
            # TODO: is it a reasonable way to score?
            score = (tpr + tnr + tree_sparsity) / 3
        else:
            score = (tpr + tnr) / 2
        print(f"Tree score: {score}")
        return score

def fraction_predicted_hh(preds):
    return round(np.sum(preds) / preds.shape[0], 2)

def perf_pheavy(model_d, proba_thresh, test_d, five_tuple, features_stages):

    model_dict = deepcopy(model_d)
    test_dict = deepcopy(test_d)
    precision = {}
    recall = {}
    ap = {}
    tprs = {}
    tnrs = {}
    fscores = {}
    frac_hh = {}
    conf_mats = {}
    seen_preds = []
    seen_preds_proba = []
    seen_gt = []

    trees_idx = sorted(list(model_dict.keys())) # The order of insertion might be messy because of the greed search replacement process
    first_tree_pk = trees_idx[0]
    last_tree_pk = trees_idx[-1]

    # When scoring during training procedure for debug, the data structures are not the same as during testing.
    # So it requires some reshape
    if isinstance(test_dict[last_tree_pk], dict): 
        for k in test_dict.keys():
            if isinstance(test_dict[k], dict):
                test_dict[k] = test_dict[k]["test"]
            
    if isinstance(model_dict[last_tree_pk], dict):
        for k in model_dict.keys():
            model_dict[k] = model_dict[k]["tree"]

    for stage_step, pk in enumerate(trees_idx):
        if stage_step + 1 < len(trees_idx):
            next_stage = trees_idx[stage_step + 1]
        else: 
            next_stage = None
        tree = model_dict[pk]       
        X, y = test_dict[pk]
        preds_proba = tree.predict_proba(X[features_stages[pk]].values)
        if preds_proba.shape[1] == 2: 
            preds_proba = preds_proba[:, 1]
        else:
            # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
            preds_proba = preds_proba.flatten()
            print(f"Only {tree.tree_.node_count} node in decision tree.")
        preds = np.where(preds_proba <= proba_thresh, 0, 1)

        # Score as if pHeavy had no more stages
        assert np.concatenate([seen_gt, y]).shape[0] == test_dict[first_tree_pk][1].shape[0]
        assert np.concatenate([seen_preds, preds]).shape[0] == test_dict[first_tree_pk][1].shape[0]
        assert np.concatenate([seen_preds_proba, preds_proba]).shape[0] == test_dict[first_tree_pk][1].shape[0]
        precision[pk] = precision_score(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )
        recall[pk] = recall_score(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )
        ap[pk] = average_precision_score(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds_proba, preds_proba])
        )
        tprs[pk] = tpr(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )
        tnrs[pk] = tnr(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )
        fscores[pk] = f1_score(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )
        frac_hh[pk] = fraction_predicted_hh(np.concatenate([seen_preds, preds])) 
        conf_mats[pk] = confusion_matrix(
            np.concatenate([seen_gt, y]),
            np.concatenate([seen_preds, preds])
        )

        # identify the next stage samples 5 tuples according to the preds
        if pk < last_tree_pk:
            # RAM
            removable_features = []
            for p in model_dict.keys():
                if p < (next_stage):
                    removable_features.extend([f for f in features_stages[p] if f not in ["src_port", "dst_port"]])
            test_dict[next_stage] = (
                test_dict[next_stage][0].drop(columns=removable_features), 
                test_dict[next_stage][1]
            ) 

            # Identify the predicted elephants that have more the pk packets (= next stage test set)
            next_5tuple = X.reset_index(drop=True).iloc[np.where(preds == 1)][five_tuple]
            test_dict[next_stage] = (
                test_dict[next_stage][0].reset_index(drop=True), 
                test_dict[next_stage][1]
            )
            test_dict[next_stage][0]["old_index"] = test_dict[next_stage][0].index
            X_next = next_5tuple.merge(
                test_dict[next_stage][0], 
                how="left", 
                on=five_tuple, 
                indicator=True
            )
            next_idx = X_next[X_next["_merge"] == "both"]["old_index"].astype(int)
            test_dict[next_stage] = (
                test_dict[next_stage][0].iloc[next_idx].drop(columns="old_index"), 
                test_dict[next_stage][1][next_idx]
            )

            # Update preds and ground truth for next stage scores

            # 1. Add the predicted mice
            mice_preds_idx = np.where(preds == 0)[0]
            seen_gt.extend(y[mice_preds_idx].tolist())
            seen_preds.extend(preds[mice_preds_idx].tolist())
            seen_preds_proba.extend(preds_proba[mice_preds_idx].tolist())

            # 2. Add the predicted elephants that were mice (otherwise they are lost)
            X = X.reset_index(drop=True)
            X["index_init"] = X.index
            eleph_preds_smallpk_idx = X_next[X_next["_merge"] == "left_only"][five_tuple].merge(
                X[five_tuple + ["index_init"]],
                on=five_tuple,
                how="inner"
            )["index_init"].tolist()
            seen_gt.extend(y[eleph_preds_smallpk_idx].tolist())
            seen_preds.extend(preds[eleph_preds_smallpk_idx].tolist())
            seen_preds_proba.extend(preds_proba[eleph_preds_smallpk_idx].tolist())

    # RAM
    del test_dict
    del model_dict

    return (precision, recall, ap, tprs, tnrs, fscores, frac_hh, conf_mats)

def tnr(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tn = conf_mat[0][0]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    tp = conf_mat[1][1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return tnr

def tpr(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tn = conf_mat[0][0]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    tp = conf_mat[1][1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return tpr

def train_pheavy(pk, X_train, y_train, features_pk, cost_sensitive, max_d, m, n):
    # print(f"\n# Training tree for the {pk}-th packet arrival")

    # Re-sampling
    if np.unique(y_train).shape[0] == 2:
        single_class = False
        oss_sampler = OneSidedSelection()
        X_train_oss, y_train_oss = oss_sampler.fit_resample(X_train[features_pk], y_train)
        undersampler = RandomUnderSampler(sampling_strategy='auto', replacement=False)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_oss[features_pk], y_train_oss)
    else:
        # Cornercase: when the previous trees have managed to keep only a single class in the train set at this stage
        # Then, re-sampling would throw an error
        single_class = True
        X_train_resampled, y_train_resampled = X_train[features_pk], y_train

    # Cost-sensitive learning
    if cost_sensitive and not single_class:
        # Based on tuning experiments to obtain high TPR and high TNR like pHeavy
        cost = np.array([
            [0, 2.5], 
            [1, 0],
        ])
        # TODO: cost = COSTS[pk] tuned for each stage
        S = X_train_resampled
        S["target"] = y_train_resampled
        print(f"Performing MetaCost with {m} resampling sets and FN penalty {cost[0][1]}")
        dt = MetaCost(
            S=S, L=DecisionTreeClassifier(max_depth=max_d), C=cost, 
            m=m, n=n, p=True, q=True
        )
        dt = dt.fit(flag='target', num_class=2)
    else:
        dt = DecisionTreeClassifier(max_depth=max_d).fit(X_train_resampled.values, y_train_resampled)

    return dt

def print_missing(X_train_next, X_test_next, next_5tuple_train, next_5tuple_test):
    missing_next_train = X_train_next[X_train_next["_merge"] == "left_only"] # Train flows with more than $pk packets, that have been predicted as mice
    missing_next_test = X_test_next[X_test_next["_merge"] == "left_only"] # Test flows with more than $pk packets, that have been predicted as mice
    print(f"Missing in next train: {missing_next_train.shape[0] / next_5tuple_train.shape[0]}")
    print(f"Missing in next test: {missing_next_test.shape[0] / next_5tuple_test.shape[0]}")

def get_data_for_next_stage(dt, pk, next_stage, features_pk, five_tuple, max_pk, proba_thresh, X_train, X_test, data):
    preds_train = dt.predict_proba(X_train[features_pk].values)
    if preds_train.shape[1] == 2: 
        preds_train = preds_train[:, 1]
    else:
        # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
        preds_train = preds_train.flatten()
        print(f"Only {dt.tree_.node_count} node in decision tree.")
    preds_train = np.where(preds_train <= proba_thresh, 0, 1)

    preds_test = dt.predict_proba(X_test[features_pk].values)
    if preds_test.shape[1] == 2: 
        preds_test = preds_test[:, 1]
    else:
        # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
        preds_test = preds_test.flatten()
        print(f"Only {dt.tree_.node_count} node in decision tree.")
    preds_test = np.where(preds_test <= proba_thresh, 0, 1)

    next_stage_train = (None, None)
    next_stage_test = (None, None)
    if next_stage < max_pk:
        # Identify the predicted elephants that have more the pk packets (= next stage data sets)
        next_5tuple_train = X_train.reset_index(drop=True).iloc[np.where(preds_train == 1)][five_tuple]
        next_5tuple_test = X_test.reset_index(drop=True).iloc[np.where(preds_test == 1)][five_tuple]
        next_data = (
            data[next_stage][0].reset_index(drop=True), 
            data[next_stage][1]
        )
        next_data[0]["old_index"] = next_data[0].index
        X_train_next = next_5tuple_train.merge(
            next_data[0], 
            how="left", 
            on=five_tuple, 
            indicator=True
        )
        X_test_next = next_5tuple_test.merge(
            next_data[0], 
            how="left", 
            on=five_tuple, 
            indicator=True
        )

        correct_next_train = X_train_next[X_train_next["_merge"] == "both"]["old_index"].astype(int)
        correct_next_test = X_test_next[X_test_next["_merge"] == "both"]["old_index"].astype(int)
        
        next_stage_train = (next_data[0].iloc[correct_next_train].drop(columns="old_index"), next_data[1][correct_next_train])
        next_stage_test = (next_data[0].iloc[correct_next_test].drop(columns="old_index"), next_data[1][correct_next_test])
        # print_missing(X_train_next, X_test_next, next_5tuple_train, next_5tuple_test)

    return next_stage_train, next_stage_test

def thr_linear_search(eleph_tracker_nentries, X_df, preds_proba, share_evicted=None):

    # Find thresholds by linear search
    model_thresholds = {}
    for eleph_nentries in eleph_tracker_nentries:
        if share_evicted:
            # Some flows will not reach the model from the Flow Manager
            preds_proba_evicted = deepcopy(preds_proba)
            n_flows_kickedout_pkc = int(share_evicted * preds_proba_evicted.shape[0]) 
            indices = np.random.choice(preds_proba_evicted.shape[0], size=n_flows_kickedout_pkc, replace=False)
            preds_proba_evicted[indices] = 0.
        else:
            preds_proba_evicted = preds_proba
        tolerance = int(eleph_nentries * 0.05)
        current_thr = 1.
        prev_thr = 1.5
        maximum = 1.
        minimum = 0.
        nelephs = -np.inf
        diff_thr = +np.inf # Avoid infinite loop when the ideal thr does not exist

        while abs(nelephs - eleph_nentries) > tolerance and diff_thr > 0.0005:
            tmp = current_thr
            if nelephs < eleph_nentries:
                # Reduce thr to increase the number of predicted elephs
                maximum = current_thr
                current_thr = (current_thr + minimum) / 2.
            else:
                # Increase thr to reduce the number of predicted elephs
                minimum = current_thr
                current_thr = (current_thr + maximum) / 2.
            prev_thr = tmp
            diff_thr = abs(prev_thr - current_thr)

            # Estimate the output preds using current_thr and voting
            preds_proba_cp = deepcopy(preds_proba_evicted)
            preds_proba_cp = np.where(preds_proba_cp < current_thr, 0., 1.)
            current_preds = preds_proba_cp

            # Compute the number of elephs using current_thr
            nelephs = np.count_nonzero(current_preds)

        print(f"Using thr {current_thr} to output {nelephs} elephants.")
        model_thresholds[eleph_nentries] = current_thr 

    return model_thresholds

def get_threshold(
    model,
    n_packets: int,
    pct_hh: list,
    minute_paths: list,
    share_evicted: float=None,
    features=None,
    dry_run: int=0,
    eleph_tracker_nentries: list=[],
    percentile=99,
):
    # Parse data
    data_proba_thr = parse_traces_pheavy(minute_paths, features, n_pk_min=n_packets, n_pk_max=n_packets+1, dry_run=dry_run)
    (X, y) = data_proba_thr[n_packets]

    # Get probability output from the model trees
    preds_proba = model.predict_proba(X)[:, 1]

    # Compute proba thr 
    model_thresholds = thr_linear_search(eleph_tracker_nentries, X, preds_proba, share_evicted)

    return model_thresholds

def main(args):

    max_d = args.max_depth
    min_pk = args.min_pk
    max_pk = args.max_pk
    proba_thresh = 0.5
    features_pheavy = []
    features_stages = {}
    for i in range(args.min_pk, args.max_pk + 1):
        pk_feat = [
            f"{f}_{i+1}" 
            if ("pk_size" in f) else f"{f}_{i}" 
            for f
            in args.features_pheavy
        ]
        features_stages[i] = ["src_port", "dst_port"] + pk_feat
        features_pheavy.extend(pk_feat)

    # Load data
    print(f"Training on {len(args.train_init)} minutes and testing on {len(args.test_traces)} minutes.")
    five_tuple = ["src_ip","dst_ip","src_port","dst_port","protocol", "first_ts"] # We add first_ts because there may be duplicate 5-tuples in the same minute
    data = parse_traces_pheavy(
        args.train_init, five_tuple + features_pheavy, n_pk_min=args.min_pk, n_pk_max=args.max_pk, dry_run=args.dry_run, percentile=args.percentile,
    )
    
    # Greedy tree selection algorithm
    selected_idx = []
    pheavy_selected = []
    score_thr = args.thr # TODO: check how to init
    n_trees = args.n_trees
    start_search = min_pk
    end_search = max_pk
    search_pace_delta = args.pace_delta
    current_tree_pk = start_search - 1
    X, y = data[start_search]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    test_data = deepcopy(data)
    test_data[start_search] = (X_test, y_test) # We evaluate only on the test set
    
    while (len(pheavy_selected) < n_trees) and (current_tree_pk < end_search) and (X_train is not None):

        print(f"\n### Greedy search for {len(pheavy_selected) + 1} / {n_trees} total trees")

        # Find the next tree that satisfies greedy threshold
        X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = X_train, y_train, X_test, y_test 
        next_idx = current_tree_pk
        score_tmp = 0
        while (score_tmp < score_thr) and (next_idx < end_search) and (X_train_tmp is not None):
            next_idx = next_idx + 1
            print(f"Training at the {next_idx}-th pk (init loop)")
            features_pk = features_stages[next_idx]
            next_tree = train_pheavy(
                next_idx, 
                X_train_tmp, 
                y_train_tmp, 
                features_pk, 
                cost_sensitive=True,
                max_d=max_d, 
                m=args.m, 
                n=args.n
            )
            perf = perf_pheavy(
                model_d={k:v for k,v in zip(selected_idx + [next_idx], pheavy_selected + [next_tree])}, 
                proba_thresh=proba_thresh, 
                # test_d=data,
                test_d=test_data,
                five_tuple=five_tuple, 
                features_stages=features_stages
            )
            print(f"{perf=}") # (precision, recall, ap, tpr, tnr, fscore, frac_hh, conf_mat)
            # score_tmp = score_greedysearch(
            #     dt=next_tree, 
            #     X_test=X_test_tmp[features_pk], 
            #     y_test=y_test_tmp, 
            #     max_depth=max_d
            # )
            tpr_tmp = perf[3][next_idx]
            tnr_tmp = perf[4][next_idx]
            score_tmp = (tpr_tmp + tnr_tmp) / 2

            if next_idx == args.initial_stage:
                break

            (X_train_tmp, y_train_tmp), (X_test_tmp, y_test_tmp) = get_data_for_next_stage(
                dt=pheavy_selected[-1], 
                pk=selected_idx[-1], 
                next_stage=next_idx + 1,
                features_pk=features_stages[selected_idx[-1]], 
                five_tuple=five_tuple, 
                max_pk=max_pk, 
                proba_thresh=proba_thresh, 
                X_train=X_train,
                X_test=X_test, 
                data=data,
            )

        # Add the tree that has satisfied greedy threshold to pHeavy pipeline
        pheavy_selected.append(next_tree)
        selected_idx.append(next_idx)
        current_tree_pk = next_idx
        print(f"- Satisfying tree found at {current_tree_pk} pk")

        # Check if a better tree can be found next up in the pace delta
        if next_idx > args.initial_stage:
            delta = search_pace_delta
            while (delta > 0) and (next_idx < end_search) and (X_train_tmp is not None):
                next_idx = next_idx + 1
                print(f"Training at the {next_idx}-th pk (pace delta loop)")
                features_pk = features_stages[next_idx]
                next_tree_delta = train_pheavy(
                    next_idx, 
                    X_train_tmp, 
                    y_train_tmp, 
                    features_pk, 
                    cost_sensitive=True, 
                    max_d=max_d,
                    m=args.m, 
                    n=args.n
                )
                perf_delta = perf_pheavy(
                    model_d={k:v for k,v in zip(selected_idx[:-1] + [next_idx], pheavy_selected[:-1] + [next_tree_delta])}, 
                    proba_thresh=proba_thresh, 
                    # test_d=data,
                    test_d=test_data,
                    five_tuple=five_tuple, 
                    features_stages=features_stages
                )
                print(f"{perf_delta=}") # (precision, recall, ap, tpr, tnr, fscore, frac_hh, conf_mat)
                # score_delta = score_greedysearch(
                #     dt=next_tree_delta, 
                #     X_test=X_test_tmp[features_pk], 
                #     y_test=y_test_tmp, 
                #     max_depth=max_d
                # )
                tpr_delta = perf_delta[3][next_idx]
                tnr_delta = perf_delta[4][next_idx]
                score_delta = (tpr_delta + tnr_delta) / 2
                if score_delta > score_tmp:
                    score_tmp = score_delta
                    pheavy_selected[-1] = next_tree_delta
                    selected_idx[-1] = next_idx
                    print(f"- Replaced last tree by: {next_idx}")
                current_tree_pk = next_idx
                delta = delta - 1
                if (delta > 0) and (next_idx < end_search):
                    (X_train_tmp, y_train_tmp), (X_test_tmp, y_test_tmp) = get_data_for_next_stage(
                        dt=pheavy_selected[-2], 
                        pk=selected_idx[-2], 
                        next_stage=next_idx + 1,
                        features_pk=features_stages[selected_idx[-1]], 
                        five_tuple=five_tuple, 
                        max_pk=max_pk, 
                        proba_thresh=proba_thresh, 
                        X_train=X_train,
                        X_test=X_test, 
                        data=data,
                    )

        # Generate data for the next init_iter
        (X_train, y_train), (X_test, y_test) = get_data_for_next_stage(
            dt=pheavy_selected[-1], 
            pk=selected_idx[-1], 
            next_stage=next_idx + 1,
            features_pk=features_stages[selected_idx[-1]], 
            five_tuple=five_tuple, 
            max_pk=max_pk, 
            proba_thresh=proba_thresh, 
            X_train=X_train,
            X_test=X_test, 
            data=data,
        )
        print(f"### End of init_loop {len(pheavy_selected)} / {n_trees}. Selected trees: {selected_idx}")

    print(f"Final list of trees: {selected_idx}")

    # Size pHeavy
    sizes = size_pheavy(
        trees_list=pheavy_selected, 
        n_b_feat_idx=args.n_b_feat_idx, # TODO: unfair because not all features require 16 bits (e.g., pk_size_max)
        w=args.w
    )
    print(f"{sizes=}")

    # Save
    pheavy_dict = {}
    for pk, tree_dict, s in zip(selected_idx, pheavy_selected, sizes):
        pheavy_dict[pk] = {
            "tree": tree_dict,
            "size": s["size"],
        }
    with open(f"{args.save_folder}model_pheavy.pkl", "wb") as f:
        pickle.dump(pheavy_dict, f)

    # Define probability threshold (i.e., when enforcing an Eleph Tracker max size)
    # Only used with the pHeavy stage regarding the 5th packet arrival 
    # Only used for fair comparison against DUMBO in the use case benchmark 
    proba_thr_eleph_tr = get_threshold(
        model=pheavy_dict[5]["tree"],
        n_packets=5,
        pct_hh=[args.share_predicted_hh],
        minute_paths=[args.train_init[-1]],
        share_evicted=args.share_evicted,
        # features=five_tuple + features_pheavy,
        features=features_stages[5],
        dry_run=args.dry_run,
        eleph_tracker_nentries=[] if args.eleph_tracker_nentries == 0 else [args.eleph_tracker_nentries],
        percentile=args.percentile,
    )[args.eleph_tracker_nentries if args.eleph_tracker_nentries > 0 else args.share_predicted_hh]

    # Testing on new minutes
    precisions = []
    recalls = []
    ap_scores = []
    tprs = []
    tnrs = []
    f1_scores = []
    conf_mats = []
    conf_mats_proba_thr = []
    for minute, trace in enumerate(args.test_traces):
        print(f"\n######### Minute {minute} ###########")
        
        # Load data
        test_dict = parse_traces_pheavy(
            [trace], five_tuple + features_pheavy, n_pk_min=args.min_pk, n_pk_max=args.max_pk, dry_run=args.dry_run
        )

        # Compute scores for init and continual models
        (precision, recall, ap, tpr, tnr, f1, frac_hh, conf_mat) = perf_pheavy(pheavy_dict, proba_thresh, test_dict, five_tuple, features_stages)
        print(f"Precision Init: {precision}")
        print(f"Recall Init: {recall}")
        print(f"AP score Init: {ap}")
        print(f"TPR Init: {tpr}")
        print(f"TNR Init: {tnr}")
        print(f"F1 score Init: {f1}")
        print(f"Fraction predicted as HH: {frac_hh}")
        precisions.append(precision)
        recalls.append(recall)
        ap_scores.append(ap)
        tprs.append(tpr)
        tnrs.append(tnr)
        f1_scores.append(f1)
        conf_mats.append(conf_mat)

        # Compute confusion matrix when imposing an Eleph Tracker max size (i.e., tune a proba thr).
        # Only for pHeavy 5th packet stage 
        # Only saved for fair benchmarking on use cases.
        tree_prob_thr = pheavy_dict[5]["tree"]       
        X_prob_thr, y_prob_thr = test_dict[5]
        preds_proba_prob_thr = tree_prob_thr.predict_proba(X_prob_thr[features_stages[5]].values)
        if preds_proba_prob_thr.shape[1] == 2: 
            preds_proba_prob_thr = preds_proba_prob_thr[:, 1]
        else:
            preds_proba_prob_thr = preds_proba_prob_thr.flatten() # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
            print(f"Only {preds_proba_prob_thr.tree_.node_count} node in decision tree.")
        preds_thresholded = np.where(preds_proba_prob_thr <= proba_thr_eleph_tr, 0, 1)
        conf_mat_proba_thr = confusion_matrix(y_prob_thr, preds_thresholded)
        conf_mats_proba_thr.append(conf_mat_proba_thr)

    # Save score
    with open(args.save_folder + f"minute_APscore_initial_vs_CL.pkl", "wb") as f:
        pickle.dump(
            {
                "initial_model_Precision": precisions,
                "initial_model_Recall": recalls,
                "initial_model_AP": ap_scores,
                "initial_model_TPR": tprs,
                "initial_model_TNR": tnrs,
                "initial_model_F1": f1_scores,
                "initial_model_conf_mats": conf_mats,
                "initial_model_conf_mats_proba_thr": conf_mats_proba_thr,
            }, 
            f
        )
    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--percentile", help="Elephants definition", type=float, default=99)
    argparser.add_argument("--train-init", nargs="+", help="Preprocessed initial training traces CSV", type=str, required=True)
    argparser.add_argument("--test-traces", nargs="+", help="Preprocessed test traces CSV", type=str, required=True)
    argparser.add_argument("--min-pk", help="Minimum number of packets when loading the traces data", type=int, default=5)
    argparser.add_argument("--max-pk", help="Maximum number of packets when loading the traces data", type=int, default=20)
    argparser.add_argument("--features-pheavy", nargs="+", help="Feature names to parse from the oracle", type=str, required=True)
    argparser.add_argument("--max-depth", help="Tree maximum depth", type=int, default=10)
    argparser.add_argument("--n-trees", help="Total number of trees for pHeavy", type=int, default=4)
    argparser.add_argument("--start-search", help="n-th packet arrival at which we can start to train a decisoin tree", type=int, default=5)
    argparser.add_argument("--end-search", help="n-th packet arrival at which we can stop to train a decisoin tree", type=int, default=20)
    argparser.add_argument("--pace-delta", help="Pace delta to look for better trees in the next packets in the greedy search algo ", type=int, default=5)
    argparser.add_argument("--thr", help="Score threshold for the greedy search algo", type=float, default=0.7)
    argparser.add_argument("--cost-sensitive", help="Use a cost-sensitivy matrix populated with heuristics on class imbalance", action="store_true")
    argparser.add_argument("--rs", help="Find each tree hyperparameters through random search", action="store_true")
    argparser.add_argument("--hyperparams", help="Hyperparams search space JSON file", type=str, required=False)
    argparser.add_argument("--m", help="Number of resample sets in MetaCost", type=int, default=10)
    argparser.add_argument("--n", help="Fraction of examples in each resample of MetaCost", type=float, default=0.5)
    argparser.add_argument("--w", help="Number of bits to represent a floating point feature", type=int, default=16)
    argparser.add_argument("--n-b-feat-idx",  nargs="+", help="Indices of the floating point features (used for sizing)", type=int, default=10)
    argparser.add_argument("--initial-stage",  help="Index of the first (forced) stage for pHeavy", type=int, default=5)
    argparser.add_argument("--dry-run", type=int, help="Keep only the first n instances in each minute", default=0, required=False)
    argparser.add_argument("--save-folder", help="Saving the model", type=str, required=True)

    argparser.add_argument("--eleph-tracker-nentries", help="Number of keys that can be stored in the Elephant Tracker HT. If this args is given, then --share-predicted-hh is ignored", type=int, default=0)
    argparser.add_argument("--share-evicted", help="Share of Elephant evicted from the Flow Manager (used for defining the proba thresh)", type=float, required=False, default=0.29725871916541285) # Default for CAIDA1
    argparser.add_argument("--share-predicted-hh", help="Share of all the minute flows (> 1pk) that should be predicted as HH. Used to determine proba thresh", type=float)

    args = argparser.parse_args()
    
    with open(args.save_folder + "args_train_val_continual_voting_pipeline.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
              
    main(args)
    