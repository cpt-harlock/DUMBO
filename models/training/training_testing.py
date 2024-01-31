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

import pickle
import numpy as np
import pandas as pd
import random
from scipy.stats import uniform # used for random search space
from statistics import fmean

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from copy import deepcopy
import random
import time
import argparse
import json
from copy import deepcopy
from pprint import pprint
from multiprocess import Pool
from functools import partial

SEED = 42
random.seed(SEED)
np.random.seed(SEED)   


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
    X, y, y_sizes, cutoffs = get_X_y_binary_size_cutoffs(
        filenames=minute_paths, 
        feature_names=features, 
        min_packets=n_packets,
        percentile=percentile,
        verbose=True,
        dry_run=dry_run,
    )
        
    # Get probability output from the model trees
    X_df = pd.DataFrame(data=X, columns=features)
    for name, step in model.named_steps.items():
        if name == "model":
            preds_proba = step.predict_proba_trees(X_df)
        else:
            X_df = step.transform(X_df)

    model_thresholds = thr_linear_search(eleph_tracker_nentries, X_df, preds_proba, share_evicted)

    return model_thresholds

def thr_linear_search(eleph_tracker_nentries, X_df, preds_proba, share_evicted=None):

    # NB: Shape of preds_proba (trees votes): (n_trees, n_samples)
    # Find thresholds by linear search
    model_thresholds = {}
    for eleph_nentries in eleph_tracker_nentries:
        if share_evicted:
            # Some flows will not reach the model from the Flow Manager
            preds_proba_evicted = deepcopy(preds_proba)
            n_flows_kickedout_pkc = int(share_evicted * preds_proba_evicted.shape[1]) 
            indices = np.random.choice(preds_proba_evicted.shape[1], size=n_flows_kickedout_pkc, replace=False)
            preds_proba_evicted[:, indices] = 0.
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
            current_preds = preds_proba_cp.mean(axis=0) # Shape: (n_trees, n_samples)
            current_preds = np.where(current_preds < 0.5, 0, 1) # If there is a tie, we favor the positive class

            # Compute the number of elephs using current_thr
            nelephs = np.count_nonzero(current_preds)

        print(f"Using thr {current_thr} to output {nelephs} elephants.")
        model_thresholds[eleph_nentries] = current_thr 

    return model_thresholds

def simulate_threshold(
    model,
    pct_hh: int,
    X_df,
    nflows_1pk,
    share_evicted: float,
    sampling_rate:float,
    eleph_tracker_nentries=[],
):

    # Get probability output from the model trees
    for name, step in model.named_steps.items():
        if name == "model":
            preds_proba = step.predict_proba_trees(X_df)
        else:
            X_df = step.transform(X_df)
    
    model_thresholds = thr_linear_search(eleph_tracker_nentries, X_df, preds_proba)

    return model_thresholds

def get_best_params_read(rs_results_csv, rs_idx=0, shrinking_level="pruning+feat_selection+quantization"):
    """Extract the params from the random search CSV"""
    rs_cv_results = pd.read_csv(rs_results_csv)
    rs_cv_results = rs_cv_results.sort_values("rank_test_score")
    best_params = rs_cv_results[rs_idx:rs_idx + 1]["params"]
    best_params = eval(best_params.values[0])
    if shrinking_level == "none":
        del best_params["feature_selection__threshold"]
        del best_params["preprocessing__kbdiscretizer__n_bins"]
    if shrinking_level == "quantization":
        del best_params["feature_selection__threshold"]
    best_val_score = rs_cv_results[rs_idx:rs_idx + 1]["mean_test_score"].values[0]
    return best_params, best_val_score

def get_best_params_cross_val(
        hyperparams_file: str, 
        shrinking_level: str,
        n_splits: int,
        n_threads: int, 
        n_iters: int,
        model, 
        X, 
        y,
        save_folder: str,
):
    """Run random search"""
    space = {}
    with open(hyperparams_file, 'rb') as f:
        params = json.load(f)
        if shrinking_level == "none":
            del params["feature_selection__threshold"]
            del params["preprocessing__kbdiscretizer__n_bins"]
        if shrinking_level == "quantization":
            del params["feature_selection__threshold"]
    for p, v in params.items():
        space[p] = eval(v)
    cv = StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True,
        random_state=SEED
    )
    rs = RandomizedSearchCV(
        estimator=model, 
        param_distributions=space, 
        n_iter=n_iters, 
        scoring='average_precision', 
        n_jobs=n_threads, 
        cv=cv, 
        random_state=SEED,
        refit=False,
        return_train_score=True,
        verbose=1,
    )
    rs.fit(X, y)
    rs_results = pd.DataFrame(rs.cv_results_)
    rs_results.to_csv(save_folder + "rs_cv_results.csv")
    rs_results = rs_results.sort_values("rank_test_score")
    best_params = rs_results[0:1]["params"]
    best_params = best_params.values[0]
    best_val_score = rs_results[0:1]["mean_test_score"].values[0]

    return best_params, best_val_score
    
def get_data(train, features, n_pk, dry_run, percentile=99):
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
   
def get_params(rs_results, X, y, n_iters, n_splits, hyperparams, shrinking_level, save_folder, n_pk, n_jobs, seed):
    if rs_results != "0":
        params, val_score = get_best_params_read(rs_results, rs_idx=0, shrinking_level=shrinking_level)
    else:
        params, val_score = get_best_params_cross_val(
            hyperparams_file=hyperparams,
            shrinking_level=shrinking_level,
            n_splits=n_splits,
            n_threads=10, 
            n_iters=n_iters,
            model=get_pipeline(params=None, n_pk=n_pk, n_jobs=int(n_jobs/10), shrinking_level=shrinking_level, seed=seed),
            X=X, 
            y=y,
            save_folder=save_folder,
        )
    return params, val_score

def get_pipeline(params, n_pk, n_jobs, shrinking_level="none", seed=SEED):
    """Get pipeline initialized with params"""
    pipeline = Pipeline(
        steps=[  
            (
                "feature_selection",
                SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=20,
                        max_depth=10,
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                ).set_output(transform='pandas')
            ),
            (
                "preprocessing", 
                ColumnTransformer(
                    transformers=[(
                        'kbdiscretizer', 
                        KBinsDiscretizer(
                            encode='onehot-dense', 
                            strategy='quantile', 
                            dtype=None, 
                            subsample=100_000,
                            random_state=SEED
                        ), 
                        [
                            "pk_size_mean_"+str(n_pk+1), 
                            "pk_size_std_"+str(n_pk+1), 
                            "iat_mean_"+str(n_pk), 
                            "iat_std_"+str(n_pk)
                        ] if n_pk >= 2 else []
                        # ] if n_pk > 2 else [
                        #     "pk_size_mean_"+str(n_pk+1), 
                        #     "pk_size_std_"+str(n_pk+1), 
                        #     "iat_mean_"+str(n_pk)
                        # ]
                    )],
                    verbose_feature_names_out=False,
                    remainder="passthrough"
                )
            ),
            (
                "useless_feat_removal",
                VarianceThreshold()
            ),
            (
                "model", 
                VotingRandomForestClassifier(
                    n_jobs=n_jobs,
                    random_state=seed
                )
            )
        ]
    )

    if shrinking_level == "none":
        pipeline = Pipeline(deepcopy(pipeline).steps[3:])

    if shrinking_level == "quantization":
        pipeline = Pipeline(deepcopy(pipeline).steps[1:])

    if (shrinking_level == "feat_selection+quantization") or (shrinking_level == "pruning+feat_selection+quantization") or (shrinking_level == "aggressivepruning+feat_selection+quantization"):
        pass

    if params:
        pipeline.set_params(**params)

    return pipeline

def manage_buffers(
    model,
    X_buffer, y_buffer,
    X_init, y_init,
    init_set_handling,
    max_retraining_set_size=0,
):
    if init_set_handling == "keep":
        # Always keep the initial train set
        X_retrain = pd.concat([X_init] + X_buffer)
        y_retrain = np.concatenate([y_init] + y_buffer)

    if init_set_handling == "add":
        # Growing init set
        X_retrain = pd.concat([X_init] + X_buffer)
        y_retrain = np.concatenate([y_init] + y_buffer)
        X_init = X_retrain
        y_init = y_retrain

    if init_set_handling == "add-fifo":
        # Growing init set up to a certain size
        X_retrain = pd.concat([X_init] + X_buffer)
        y_retrain = np.concatenate([y_init] + y_buffer)
        X_init = X_retrain[-max_retraining_set_size:]
        y_init = y_retrain[-max_retraining_set_size:]

    elif init_set_handling == "update-agreement":
        # Keep the hardest examples in the initial train set
        X_retrain = pd.concat([X_init] + X_buffer)
        y_retrain = np.concatenate([y_init] + y_buffer)
        agreements = get_agreements(X_retrain, model)
        sorted_agreements = np.argsort(agreements)
        X_init = X_retrain.iloc[sorted_agreements[:X_init.shape[0]]]
        y_init = y_retrain[sorted_agreements[:X_init.shape[0]]]

    elif init_set_handling == "update-fifo":
        # Keep the newest examples in the initial train set
        X_retrain = pd.concat([X_init] + X_buffer)
        y_retrain = np.concatenate([y_init] + y_buffer)
        n_new = pd.concat(X_buffer).shape[0]
        X_init = X_retrain[n_new:]
        y_init = y_retrain[n_new:]

    elif init_set_handling == "discard":
        # Forget about the initial train set
        X_retrain = pd.concat(X_buffer)
        y_retrain = np.concatenate(y_buffer)

    print(f"{pd.concat(X_buffer).shape[0]=}")
    print(f"{X_init.shape[0]=}")
    print(f"{X_retrain.shape[0]=}")
    
    return (X_retrain, y_retrain), ([], []), (X_init, y_init)

def parse_traces(init_traces, continual_traces, features, n_pk, dry_run, percentile=99):
    print(f"Elephants are defined with the {percentile}-th percentile.")
    nflows_1pk = {}
    data = {}

    # Init traces
    X_init, y_init = get_data(init_traces, features, n_pk, dry_run, percentile=percentile)
    start_test_minute = len(init_traces)

    # Update / test traces
    for i, minute_trace in enumerate(continual_traces):
        X, y = get_data([minute_trace], features, n_pk, dry_run)
        data[start_test_minute + i] = (X, y, minute_trace)
        X_1pk, y_1pk = get_data([minute_trace], ["src_ip"], 1, dry_run, percentile=percentile)
        nflows_1pk[start_test_minute + i] = y_1pk.shape[0]

    return X_init, y_init, data, nflows_1pk 

def populate_buffer(model_cl, sampling, min_agreement, X, y, X_buffer, y_buffer, sampling_rate=None):
    nsamples = 0
    if sampling == "active" or sampling == "both":
        # Flag remaining hard instances 
        agreements = get_agreements(X, model_cl)
        hard_buffer_idx = np.where(agreements < min_agreement)[0]
        X_buffer.append(X.iloc[hard_buffer_idx]), 
        y_buffer.append(y[hard_buffer_idx])
        nsamples += y[hard_buffer_idx].shape[0]

    if sampling == "random" or sampling == "both":
        n_samples = int(sampling_rate * X.shape[0])
        if sampling == "both":
            sampling_space = list(range(X.shape[0]))
            sampling_space = [i for i in sampling_space if i not in hard_buffer_idx]
            if n_samples < len(sampling_space):
                random_idx = np.array(random.sample(sampling_space, n_samples))
            else:
                # Not enough samples remaining after hard-flows sampling (can happend when the pruned forest does not have enough trees to cope with the minimum agreement)
                random_idx = np.array([])
        else:
            random_idx = np.array(random.sample(range(X.shape[0]), n_samples))
        X_buffer.append(X.iloc[random_idx]), 
        y_buffer.append(y[random_idx])
        nsamples += y[random_idx].shape[0]

    print(f"Added {nsamples} samples to buffer")

    return X_buffer, y_buffer, nsamples

def scores(pipeline_init, pipeline_cl, proba_thresh_init, proba_thresh_cl, X, y, nflows_1pk, minute):
    # Preds
    # TODO: check that the `thresh` arg is passed on to the VotingRandomForestClassifier in the pipeline when shrinking
    preds_init = pipeline_init.predict(X, thresh=proba_thresh_init)
    preds_cl = pipeline_cl.predict(X, thresh=proba_thresh_cl)

    # Precision
    precision_init = precision_score(y, preds_init)
    precision_cl = precision_score(y, preds_cl)

    # Recall
    recall_init = recall_score(y, preds_init)
    recall_cl = recall_score(y, preds_cl)

    # Average precision score
    preds_proba_init = pipeline_init.predict_proba(X)
    if preds_proba_init.shape[1] == 2: 
        preds_proba_init = preds_proba_init[:, 1]
    else:
        # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
        preds_proba_init = preds_proba_init.flatten()
    preds_proba_cl = pipeline_cl.predict_proba(X)
    if preds_proba_cl.shape[1] == 2: 
        preds_proba_cl = preds_proba_cl[:, 1]
    else:
        # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
        preds_proba_cl = preds_proba_cl.flatten()
    ap_init = average_precision_score(y, preds_proba_init)
    ap_cl = average_precision_score(y, preds_proba_cl)

    # F1
    f1_init = f1_score(y, preds_init)
    f1_cl = f1_score(y, preds_cl)

    # Confusion matrices
    conf_mat_init = confusion_matrix(y, preds_init)
    conf_mat_cl = confusion_matrix(y, preds_cl)

    # Share of predicted elephants
    share_predicted_eleph_init = np.count_nonzero(preds_init) / nflows_1pk[minute]
    share_predicted_eleph_cl = np.count_nonzero(preds_cl) / nflows_1pk[minute]

    return (precision_init, recall_init, share_predicted_eleph_init, ap_init, f1_init, conf_mat_init), (precision_cl, recall_cl, share_predicted_eleph_cl, ap_cl, f1_cl, conf_mat_cl)

def size_and_score(tree_idx, pipeline, X_val, y_val, w=0, n_b_feat_idx=[]):
    pipeline_copy = deepcopy(pipeline)
    forest = pipeline_copy["model"]
    tree = get_tree(forest, tree_idx)
    
    sizes = size(
        tree,  
        f_b=forest.n_features_in_, 
        n_b_feat_idx=n_b_feat_idx,
        # w=0, # only binary features
        w=w, 
        full_forest=forest, # Required for hybrid (N is computed among all trees)
    )
    tree_size = min(sizes.values())
    tree_size = np.ceil(tree_size / 8) / 1_000 # in KB

    pipelined_tree = deepcopy(pipeline_copy)
    pipelined_tree["model"].estimators_= [get_tree_estimator(forest, tree_idx)]
    preds_proba_tree = pipelined_tree.predict_proba(X_val)
    if preds_proba_tree.shape[1] == 2: 
        preds_proba_tree = preds_proba_tree[:, 1]
    else:
        # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
        preds_proba_tree = preds_proba_tree.flatten()
    tree_score = average_precision_score(
        y_val, 
        preds_proba_tree
    )
    res = {
        "tree": str(tree_idx),
        "size": tree_size,
        "score": tree_score,
        "impl": min(sizes, key=sizes.get),
    }
    
    return res

def shrink_pipeline(n_processes, pipeline, max_model_size, X_val, y_val,  w=0, n_b_feat_idx=[], pruning=True):
    """Reduce the pipeline RF to the best subforest respecting max size"""
    
    pipeline_copy = deepcopy(pipeline)
    forest = pipeline_copy["model"]
    
    # Size and score each tree
    trees_indices = [i for i in range(len(forest.estimators_))]
    with Pool(n_processes) as pool:  
        trees = pool.map(
            partial(
                size_and_score, 
                pipeline=pipeline_copy,
                X_val=X_val, 
                y_val=y_val,
                w=w, 
                n_b_feat_idx=n_b_feat_idx,
            ), 
            trees_indices
        )        
    original_size = sum([t["size"] for t in trees])

    subforest_n_size = original_size
    if pruning:
        # Select the best subforest respecting max size
        subforest_n_size = 0
        n = 0
        
        while (n < len(trees) - 1):
            n = n + 1
            best_trees_idx = get_best_trees_idx(trees, n)
            pipelined_subforest = deepcopy(pipeline_copy)
            pipelined_subforest["model"].estimators_ = get_subforest_estimators(forest, best_trees_idx)

            # Note: here we potentilly mix various implementations in the subforest...
            subforest_n_size = sum([trees[i]["size"] for i in best_trees_idx])
                
            # Remove unused trees from the pipeline model
            subforest = [
                t 
                for i, t in enumerate(pipeline["model"].estimators_)
                if i in best_trees_idx
            ]

            # Break if size reached
            condition = (subforest_n_size > max_model_size and n >= 2)  # at least 2 trees in the subforest to avoid ties
            if max_model_size < 500:
                condition = (subforest_n_size > max_model_size)  # aggressive pruning can allow for less than 2trees
            if condition:
                break

        pipeline["model"].estimators_ = subforest
    
    return pipeline, subforest_n_size, original_size

def train_shrink_pipeline(
        X, y, init_traces, features, params, n_pk, 
        n_jobs, max_model_size, share_predicted_hh, do_save, save_folder, 
        seed=SEED, proba_thresh=None, cl_minute=0, w=0, n_b_feat_idx=[], shrinking_level="pruning+feat_selection+quantization", 
        dry_run=0, share_evicted=0.29725871916541285, eleph_tracker_nentries=0, percentile=99,
    ):
    
    # Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(f"Training with {X.shape[0]} examples.")
    print(f"Validating with {X_test.shape[0]} examples.")
    print(f"Class imbalance in train set: {np.count_nonzero(y) / y.shape[0]}")
    
    # Model
    pipeline = get_pipeline(params=params, n_pk=n_pk, n_jobs=int(n_jobs/10), seed=seed, shrinking_level=shrinking_level)
    # pipeline = pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)

    # Shrink
    if "aggressivepruning" in shrinking_level:
        max_size = 100
    elif "ultrapruning" in shrinking_level:
        max_size = 50
    else:
        max_size = max_model_size
    shrunk_pipeline, shrunk_pipeline_size, original_pipeline_size = shrink_pipeline(
        n_jobs,
        pipeline, 
        max_model_size=max_size, 
        X_val=X_test,
        y_val=y_test,
        w=w, 
        n_b_feat_idx=n_b_feat_idx,
        pruning=True if ("pruning" in shrinking_level) else False
    )
    print(f"Shrinking: {shrinking_level}")
    print(f"Pipeline size: {shrunk_pipeline_size}.")
    print(f"Pipeline ntrees: {len(shrunk_pipeline['model'].estimators_)}.")

    # probability threshold
    if not proba_thresh:
        proba_thresh = get_threshold(
            model=pipeline,
            n_packets=n_pk,
            pct_hh=[share_predicted_hh],
            minute_paths=[init_traces[-1]],
            share_evicted=share_evicted,
            features=features,
            dry_run=dry_run,
            eleph_tracker_nentries=[] if eleph_tracker_nentries == 0 else [eleph_tracker_nentries],
            percentile=percentile,
        )[eleph_tracker_nentries if eleph_tracker_nentries > 0 else share_predicted_hh]
        print(f"New proba thresh after update: {proba_thresh}.")
    else:
        print(f"Keeping same proba thresh after update: {proba_thresh}.")
    if do_save:
        if cl_minute:
            actual_minute = init_traces[-1].split("/")[-1].split("_")[0] # Name it after the preprocessed minute trace name
            name = f"cl_pipeline_{actual_minute}.pkl"
        else:
            name = "initial_pipeline.pkl"
        with open(save_folder + name, "wb") as f:
            pickle.dump(pipeline, f)
        with open(save_folder + name[:-4] + "_sizes.pkl", "wb") as f:
            pickle.dump({
                "original": original_pipeline_size,
                "shrunk": shrunk_pipeline_size,
                "proba_thr": proba_thresh,
                
            }, f)

    return shrunk_pipeline, proba_thresh

def get_agreements(X, model):
    # Predict each instance in the full buffer
    X_prep = False
    if "feature_selection" in list(model.named_steps.keys()):
        X_prep = model["feature_selection"].transform(X)
    if "preprocessing" in list(model.named_steps.keys()):
        if isinstance(X_prep, np.ndarray):
            X_prep = model["preprocessing"].transform(X_prep)
        else:
            X_prep = model["preprocessing"].transform(X)
    if "useless_feat_removal" in list(model.named_steps.keys()):
        if isinstance(X_prep, np.ndarray):
            X_prep = model["useless_feat_removal"].transform(X_prep)
        else:
            X_prep = model["useless_feat_removal"].transform(X)
    if not isinstance(X_prep, np.ndarray):
        # No shrinking
        X_prep = X
        preds = np.array([
            tree.predict(X_prep.values)
            for tree
            in model["model"].estimators_
        ])
    else:
        preds = np.array([
            tree.predict(X_prep)
            for tree
            in model["model"].estimators_
        ])

    # Compute trees agreements
    preds = np.transpose(preds) # # [ntrees, nsamples] -> [nsamples, ntrees]
    preds = np.sort(preds, axis=1)
    neg_votes = preds.argmax(axis=1) + 1 #i.e., how many trees predicted 0 before the first that predicted 1 (sorted)
    pos_votes = preds.shape[1] - neg_votes
    diff = np.abs(pos_votes - neg_votes)
    agreements = diff / len(model["model"].estimators_) * 1.0

    return agreements


def main(args):

    # Load data
    with open(args.features) as f:
        features = [line.rstrip() for line in f]
    args.features = features
    if args.n_pk == 1:
        args.features = [f for f in args.features if ("mean" not in f) and ("std" not in f)]
        args.share_evicted = 0.0
        args.w = 0
        args.n_b_feat_idx = []
        print(args.features)
    if "quantization" in args.shrinking_level:
        args.w = 0
        args.n_b_feat_idx = []

    print(f"Training on {len(args.train_init)} minutes and testing on {len(args.train_update)} minutes.")
    X_init, y_init, data, nflows_1pk = parse_traces(args.train_init, args.train_update, args.features, args.n_pk, args.dry_run, args.percentile)

    # Train initial model
    print(f"Init train dataset: {X_init.shape}")
    params_init, val_score_init = get_params(args.rs_results, X_init, y_init, args.n_iters, args.n_splits, args.hyperparams, args.shrinking_level, args.save_folder, args.n_pk, args.n_jobs, seed=SEED)
    print(f"Best params with score {val_score_init} from random search {args.rs_results}: \n {params_init}")
    if args.save_models:
        with open(args.save_folder + f"best_params.pkl", "wb") as f:
            pickle.dump(params_init, f)
             
    pipeline_init, proba_thresh_init = train_shrink_pipeline(
        X_init, y_init, args.train_init, args.features, params_init,
        args.n_pk, args.n_jobs, args.max_model_size, args.share_predicted_hh,
        args.save_models, args.save_folder, proba_thresh=None, cl_minute=0, w=args.w, 
        n_b_feat_idx=args.n_b_feat_idx, shrinking_level=args.shrinking_level, dry_run=args.dry_run, share_evicted=args.share_evicted,
        eleph_tracker_nentries=args.eleph_tracker_nentries,
        percentile=args.percentile,
    )
    proba_thresh_cl = proba_thresh_init
    print(f"Proba thresh after epoch Init / CL: {proba_thresh_init} /  {proba_thresh_cl}.")
    proba_thresholds = [{
        "proba_thresh_init": proba_thresh_cl,
        "proba_thresh_cl": proba_thresh_init,
    }]
  
    # Continual learning
    X_buffer = []
    y_buffer = []
    pipeline_cl = deepcopy(pipeline_init)
    ap_scores_init = []
    ap_scores_cl = []
    f1_scores_init = []
    f1_scores_cl = []
    conf_mats_init = []
    conf_mats_cl = []
    cl_sampling_rates = []
    buffer_replacement_rate = []
    repl_samples = 0
    simu_drift_detector = True
    # for minute, (X, y, trace_name) in data.items():
    for enum, (minute, (X, y, trace_name)) in enumerate(data.items()):

        # TMP: Simulate drift detection -> reset init set
        if args.simu_drift_detect and minute >= args.simu_drift_detect and simu_drift_detector == True:
            print("\n///// SIMU DRIFT DETECTED //////")
            X_init, y_init = X_init[:1], y_init[:1]
            simu_drift_detector = False

        print(f"\n######### Minute {minute} ###########")

        # Compute scores for init and continual models
        (
            precision_init, 
            recall_init, 
            share_predicted_eleph_init, 
            ap_init,
            f1_init,
            conf_mat_init,
        ), (
            precision_cl, 
            recall_cl, 
            share_predicted_eleph_cl, 
            ap_cl,
            f1_cl,
            conf_mat_cl
        ) = scores(pipeline_init, pipeline_cl, proba_thresh_init, proba_thresh_cl, X, y, nflows_1pk, minute)
        print(f"Precision Init / CL: {precision_init} / {precision_cl}")
        print(f"Recall Init / CL: {recall_init} / {recall_cl}")
        print(f"AP score Init / CL: {ap_init} / {ap_cl}")
        print(f"F1 score Init / CL: {f1_init} / {f1_cl}")
        print(f"Fraction of flows (>= 1pk) predicted as Elephants Init / CL: {share_predicted_eleph_init} / {share_predicted_eleph_cl}")
        print(f"# of flows predicted as Elephants Init / CL: {conf_mat_init[:, 1].sum()} / {conf_mat_cl[:, 1].sum()}")
        ap_scores_init.append(ap_init)
        ap_scores_cl.append(ap_cl)
        f1_scores_init.append(f1_init)
        f1_scores_cl.append(f1_cl)
        conf_mats_init.append(conf_mat_init)
        conf_mats_cl.append(conf_mat_cl)

        if args.update_proba_thr_simu:
            proba_thr_init_simu = simulate_threshold(
                model=pipeline_init,
                pct_hh=[args.share_predicted_hh],
                X_df=X,
                nflows_1pk=nflows_1pk[minute],
                share_evicted=args.share_evicted,
                sampling_rate=1.0,
                eleph_tracker_nentries=[] if args.eleph_tracker_nentries == 0 else [args.eleph_tracker_nentries],
            )[args.eleph_tracker_nentries if args.eleph_tracker_nentries > 0 else args.share_predicted_hh]
            proba_thresh_init = proba_thr_init_simu
            proba_thr_cl_simu = simulate_threshold(
                model=pipeline_cl,
                pct_hh=[args.share_predicted_hh],
                X_df=X,
                nflows_1pk=nflows_1pk[minute],
                share_evicted=args.share_evicted,
                sampling_rate=1.0,
                eleph_tracker_nentries=[] if args.eleph_tracker_nentries == 0 else [args.eleph_tracker_nentries],
            )[args.eleph_tracker_nentries if args.eleph_tracker_nentries > 0 else args.share_predicted_hh]
            proba_thresh_cl = proba_thr_cl_simu
            print(f"Proba thr after minute simu. Init / CL: {proba_thresh_init} / {proba_thresh_cl}")
            proba_thresholds.append({
                "proba_thresh_init": proba_thr_init_simu,
                "proba_thresh_cl": proba_thr_cl_simu,
            })

        if args.continual:

            # Populate buffer
            X_buffer, y_buffer, nsamples = populate_buffer(pipeline_cl, args.sampling, args.min_agreement, X, y, X_buffer, y_buffer, sampling_rate=args.sampling_rate)
            sampling_rate_npk = nsamples / data[minute][1].shape[0]
            print(f"Fraction of flows (>= {args.n_pk}pk) sampled: {sampling_rate_npk}")
            sampling_rate_1pk = nsamples / nflows_1pk[minute]
            print(f"Fraction of flows (>= 1pk) sampled: {sampling_rate_1pk}")
            cl_sampling_rates.append(sampling_rate_1pk)
            repl_samples = repl_samples + nsamples
            buffer_replacement_rate.append(repl_samples / args.buffer_size)

            # Update continual model
            # TOTRY: compute drift on the most important features vs previous epoch and use it as a trigger for retraining
            # Note: we could trigger retrain only under certain conditions (e.g., drop in performance)
            # if ((minute - len(args.train_init)) % args.update_freq == 0) and minute > len(args.train_init): 
            if "uni" in trace_name:
                update_condition = (enum % args.update_freq == 0) and enum > 0
            else:
                update_condition = ((minute - len(args.train_init)) % args.update_freq == 0) and (minute > len(args.train_init))
            if update_condition: 
                
                print(f"\nUpdating model at minute {minute}")

                # Retrain on buffer
                # TODO: update proba_thresh does not work (i.e., the new threshold is too high, e.g., 0.67)
                # IDEA: adjust proba_thresh based on the HT load factor?
                start_time = time.time()
                (X_retrain, y_retrain), (X_buffer, y_buffer), (X_init, y_init) = manage_buffers(
                    pipeline_cl,
                    X_buffer, y_buffer,
                    X_init, y_init,
                    args.init_set,
                    args.max_retraining_set_size,
                ) 
                if args.update_proba_thr_full:
                # Recompute proba threshold at each update
                # Note: this is optmistic because norally we do not have access to the real traces
                    pipeline_cl_update, proba_thresh_cl = train_shrink_pipeline(
                        X_retrain, y_retrain, args.train_update[((minute-args.update_freq)-len(args.train_init))+1: (minute-len(args.train_init))+1], args.features, params_init,
                        args.n_pk, args.n_jobs, args.max_model_size, args.share_predicted_hh,
                        args.save_models, args.save_folder, proba_thresh=None, cl_minute=minute, w=args.w, 
                        n_b_feat_idx=args.n_b_feat_idx, shrinking_level=args.shrinking_level, dry_run=args.dry_run,  share_evicted=args.share_evicted,
                        eleph_tracker_nentries=args.eleph_tracker_nentries,
                        percentile=args.percentile,
                    )
                    proba_thresholds.append({"proba_thresh_cl": proba_thresh_cl,})
                else:
                    pipeline_cl_update, _ = train_shrink_pipeline(
                        X_retrain, y_retrain, args.train_init, args.features, params_init,
                        args.n_pk, args.n_jobs, args.max_model_size, args.share_predicted_hh,
                        args.save_models, args.save_folder, proba_thresh=proba_thresh_cl, cl_minute=minute,  w=args.w, 
                        n_b_feat_idx=args.n_b_feat_idx, shrinking_level=args.shrinking_level, dry_run=args.dry_run,  share_evicted=args.share_evicted,
                        eleph_tracker_nentries=args.eleph_tracker_nentries,
                        percentile=args.percentile,
                    )
                update_duration = time.time() - start_time
                print(f"- Update duration (s): {round(update_duration, 2)}")

                # Deploy updated model
                # TODO: in reality, the updated model cannot be deployed immediately        
                pipeline_cl = pipeline_cl_update

    # Save minutes scores and number of model updates
    with open(args.save_folder + f"minute_APscore_initial_vs_CL.pkl", "wb") as f:
        pickle.dump(
            {
                "initial_model_AP": ap_scores_init,
                "cl_model_AP": ap_scores_cl,
                "initial_model_F1": f1_scores_init,
                "cl_model_F1": f1_scores_cl,
                "initial_model_conf_mats": conf_mats_init,
                "cl_model_conf_mats": conf_mats_cl,
                "cl_model_sampling": cl_sampling_rates,
                "buffer_repl_rate": buffer_replacement_rate,
                "proba_thresholds": proba_thresholds,
            }, 
            f
        )
    # Note: for CODA we use minute-epoch and for MAWI we use 15-minute epoch because they feature fewer packets

    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train-init", nargs="+", help="Preprocessed initial training traces CSV", type=str, required=True)
    argparser.add_argument("--train-update", nargs="+", help="Preprocessed continual training traces CSV (that will also be used for testing)", type=str, required=True)
    argparser.add_argument("--continual", help="Perform continual learning (if False, only model scoring is performed on the train-update traces)", action="store_true")
    argparser.add_argument("--init-set", help="How to handle the initial traces in the retraining buffer", choices=["add", "add-fifo", "keep", "update-agreement", "update-fifo", "discard"], required=False)
    argparser.add_argument("--update-freq", help="Update every n epochs", type=int, required=False)
    argparser.add_argument("--buffer-size", help="Retrain buffer in number of instances", type=int, required=False, default=50_000)
    argparser.add_argument("--min-agreement", help="Mininum agreement threshold, below which an example is added to the retrain buffer", type=float, required=False)
    argparser.add_argument("--sampling", help="Sampling strategy for populating the buffer", choices=["active", "random", "both"], default="active")
    argparser.add_argument("--sampling-rate", help="Sampling rate (among flows >= n_pk) if --random-sampling", type=float, required=False, default=0.02)
    argparser.add_argument("--percentile", help="Percentile to define elephants", type=float, required=False, default=99)
    argparser.add_argument("--share-evicted", help="Share of Elephant evicted from the Flow Manager (used for defining the proba thresh)", type=float, required=False, default=0.29725871916541285) # Default for CAIDA1
    argparser.add_argument("--max-model-size", help="Maximum model size in KB", type=int, required=False, default=500)
    argparser.add_argument("--w", help="Number of bits to represent a floating point feature", type=int, required=True)
    argparser.add_argument("--n-b-feat-idx",  nargs="+", help="Indices of the floating point features (used for sizing)", type=int, default=[])
    argparser.add_argument("--shrinking-level", help="Shrinking strategies (for ablation study)", choices=["none", "quantization", "feat_selection+quantization", "pruning+feat_selection+quantization", "aggressivepruning+feat_selection+quantization", "ultrapruning+feat_selection+quantization"], default="pruning+feat_selection+quantization")
    argparser.add_argument("--share-predicted-hh", help="Share of all the minute flows (> 1pk) that should be predicted as HH. Used to determine proba thresh", type=float)
    argparser.add_argument("--eleph-tracker-nentries", help="Number of keys that can be stored in the Elephant Tracker HT. If this args is given, then --share-predicted-hh is ignored", type=int, default=0)
    argparser.add_argument("--n-pk", help="Minimum number of packets per flow (pre-filtering)", type=int, required=True, choices=[1, 5])
    argparser.add_argument("--features", help="Txt file containing feature names (for1pk or 5pk model)", type=str, required=True)
    argparser.add_argument("--rs-results", help="Saved CSV for random search results", type=str, required=False)
    argparser.add_argument("--hyperparams", help="Hyperparams search space JSON file", type=str, required=False)
    argparser.add_argument("--n-splits", help="Splits for Cross validation", type=int, required=False)
    argparser.add_argument("--n-iters", help="Nb of random search iterations", type=int, required=False)
    argparser.add_argument("--n-jobs", help="Nb of threads for the RandomSearch", type=int, required=True)
    argparser.add_argument("--simu-drift-detect", help="Minute at which the init set is reset to start fresh", type=int, default=0)
    argparser.add_argument("--max-retraining-set-size", help="Maximum number of samples on which the model is allowed to retrain (FIFO)", type=int, default=0)
    argparser.add_argument("--save-models", help="Save all models (initial and updated)", action="store_true")
    argparser.add_argument("--update-proba-thr-full", help="Recompute the probability threshold at each update step", action="store_true", default=True)
    argparser.add_argument("--update-proba-thr-simu", help="Simulate the recomputation of the probability threshold at each epoch", action="store_true", default=False)
    argparser.add_argument("--save-folder", help="Saving the model", type=str, required=True)
    argparser.add_argument("--dry-run", help="Keep only the first n instances in each minute", type=int, default=0)
   
    args = argparser.parse_args()
    
    with open(args.save_folder + "args_train_val_continual_voting_pipeline.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
              
    main(args)
    