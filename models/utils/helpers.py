import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import itertools
import argparse
import csv
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score
# from tree_based_functions import *
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings

from numpy.ma import MaskedArray
from scipy.stats import rankdata

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.model_selection._validation import _insert_error_scores
from sklearn.model_selection._validation import _normalize_score_results
from sklearn.model_selection._validation import _warn_or_raise_about_fit_failures
from sklearn.exceptions import NotFittedError
from joblib import Parallel
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import indexable, check_is_fitted, _check_fit_params
from sklearn.utils.metaestimators import available_if
from sklearn.utils.fixes import delayed
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import check_scoring
from sklearn.metrics import average_precision_score

from pprint import pprint
import pickle
from pathlib import Path
import copy
from typing import Callable, List, Tuple
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
import numpy.typing as npt
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold



def get_avg_precision(model, minute_path, features_list, n_packets, pipeline=False):
    X, y = get_X_y_binary(
        filenames=[minute_path], 
        feature_names=features_list, 
        min_packets=n_packets,
        percentile=99,
        verbose=True
    )
    assert not np.isnan(X).any()
    if pipeline:
        X_df = pd.DataFrame(
            data=X,
            columns=features_list
        )
        preds = model.predict_proba(X_df)[:, 1]
    else:
        preds = model.predict_proba(X)[:, 1]
    ap = average_precision_score(y, preds)
    
    return ap


class DeduplicatedStratifiedKFold(StratifiedKFold):
    
    def __init__(self, n_splits=5, *, shuffle=False, random_state=42, flow_ids=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.flow_ids = flow_ids
        
    def split(self, X, y=None, groups=None):
        train_array = []
        test_array = []
        
        for train, test in super().split(X, y, groups):
            train_dedup = []
            test_dedup = []
            ratio = test.shape[0] / (train.shape[0] + test.shape[0])
            
            # Remove duplicates flow IDs (from train or test, at random)
            train_flow_ids = [f for i, f in enumerate(self.flow_ids) if i in train]
            test_flow_ids = [f for i, f in enumerate(self.flow_ids) if i in test]
            duplicate_flow_ids = set(train_flow_ids).intersection(set(test_flow_ids))
            test_accept = []
            
            for s, f in zip(train, train_flow_ids):
                if f in duplicate_flow_ids:
                    if np.random.rand() < ratio:
                        train_dedup.append(s)
                    else:
                        test_accept.append(f)
                else:
                    train_dedup.append(s)
            train_array.append(train_dedup)
            
            for s, f in zip(test, test_flow_ids):
                if f in duplicate_flow_ids:
                    if f in test_accept:
                        test_dedup.append(s)
                else:
                    test_dedup.append(s)
            test_array.append(test_dedup)
            
        for train, test in zip(train_array, test_array):
            yield train, test

            
class DeduplicatedTimeSeriesSplit(TimeSeriesSplit):
    
    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0, random_state=42, flow_ids=None,):
        super().__init__(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)
        self.flow_ids = flow_ids
        
    def split(self, X, y=None, groups=None):
        train_array = []
        test_array = []
        
        for train, test in super().split(X, y, groups):
            train_dedup = []
            test_dedup = []
            ratio = test.shape[0] / (train.shape[0] + test.shape[0])

            # Remove duplicates flow IDs (from train or test, at random)
            train_flow_ids = [f for i, f in enumerate(self.flow_ids) if i in train]
            test_flow_ids = [f for i, f in enumerate(self.flow_ids) if i in test]
            duplicate_flow_ids = set(train_flow_ids).intersection(set(test_flow_ids))
            test_accept = []

            for s, f in zip(train, train_flow_ids):
                if f in duplicate_flow_ids:
                    if np.random.rand() < ratio:
                        train_dedup.append(s)
                    else:
                        test_accept.append(f)
                else:
                    train_dedup.append(s)
            train_array.append(train_dedup)

            for s, f in zip(test, test_flow_ids):
                if f in duplicate_flow_ids:
                    if f in test_accept:
                        test_dedup.append(s)
                else:
                    test_dedup.append(s)
            test_array.append(test_dedup)
            
        for train, test in zip(train_array, test_array):
            yield train, test
            
# class SampledDeduplicatedTimeSeriesSplit(TimeSeriesSplit):
    
#     def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0, sampling_r=0.1, random_state=42, flow_ids=None,):
#         super().__init__(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)
#         self.sampling_r = sampling_r
#         self.flow_ids = flow_ids
        
#     def split(self, X, y=None, groups=None):
#         train_array = []
#         test_array = []
#         for train, test in super().split(X, y, groups):
            
#             # Sample from train
#             sampled_idx = np.random.choice(
#                 range(len(train)), 
#                 int(self.sampling_r * train.shape[0]),
#                 replace=False,
#             )
#             train = [s for i, s in enumerate(train) if i in sampled_idx]
#             train_array.append(train)
            
#             # Deduplicate flow ids in test
#             train_flow_ids = [s for i, s in enumerate(self.flow_ids) if i in sampled_idx]
#             test_flow_ids = [s for i, s in enumerate(self.flow_ids) if i in test]
#             test = [s for s, f in zip(test, test_flow_ids) if f not in train_flow_ids]
#             test_array.append(test)
            
#         for train, test in zip(train_array, test_array):
#             yield train, test
            
            
def check_input(files: list[str]) -> None:
    
    for f in files:
        assert os.path.exists(f), \
        "Input minute {0} doesn't exist".format(f)
    

def _fit_and_score_and_size(
    sizing_func:Callable[[BaseEstimator, list], int], 
    sizing_func_args:dict,
    save_folder:str,
    *args,
    **kw,
) -> dict:

    """
    A re-implementation of sklearn.model_selection._validation._fit_and_score
    that saves the trained estimator and its size estimation
    """
    
    # Fit and score, returning estimator and parameters
    kw_args = copy.deepcopy(kw)
    kw_args["return_parameters"] = True
    kw_args["return_estimator"] = True
    result = _fit_and_score(
         *args, 
        **kw_args, 
    )
    
    # Save the model
    model_name = "_".join([
        str(k)+str(v) 
        for k,v 
        in result["parameters"].items()
    ])
    save_model = save_folder + "models/" + model_name
    with open(save_model + ".pkl", 'wb') as f:
        pickle.dump(result["estimator"], f)
    
    # Save model size attributes
    save_size = save_folder + "models_sizes.json"
    model_size = sizing_func(result["estimator"], *sizing_func_args)
    try:
        with open(save_size, 'rb') as f:
            sizes = json.load(f)
    except:
        sizes = {}
    sizes[model_name] = {
        "bytes": model_size,
        "trees": len(result["estimator"].estimators_),
        "max_depths": [t.tree_.max_depth for t in result["estimator"].estimators_],
        "node_count:": [t.tree_.node_count for t in result["estimator"].estimators_],
    }
    with open(save_size , 'w') as f:
        json.dump(sizes, f, indent=2)
        
    # Remove unnecessary items from result dict
    del result['estimator']
    del result['parameters']
    
    return result


class RandomizedSearchCVWithCustomSizing(RandomizedSearchCV):
    """
    A subclass of RandomizedSearchCV that includes
    model sizing and saving for each CV fit
    """
    
    def __init__(
        self,
        sizing_func: Callable[[BaseEstimator, list[int], list[int]], int],
        sizing_func_args: list[list[int]],
        save_folder: str,
        *args, 
        **kw,  
    ):
        super().__init__(*args, **kw)
        self.sizing_func = sizing_func        
        self.sizing_func_args = sizing_func_args
        self.save_folder = save_folder
    
    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.
            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.
        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                # Use custom fitting and scoring method that enables
                # saving the trained estimator and its size
                out = parallel(
                    delayed(_fit_and_score_and_size)(
                        self.sizing_func,
                        self.sizing_func_args,
                        self.save_folder,
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    
def get_X_y_groups_binary_stream_dicts(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        percentile: float,
        verbose: bool,
    ) -> tuple[list, npt.NDArray, npt.NDArray, list]:
    
    all_X = []
    all_y = []
    all_flow_groups = []
    all_minutes = []
    
    for filename in filenames:
        X, y, flow_groups = get_X_y_groups_binary(
            [filename], 
            feature_names, 
            min_packets,
            percentile,
            verbose,
        )

        X = pd.DataFrame(
            data=X,
            index=range(X.shape[0]),
            columns=feature_names
        )
        
        all_X.append(X)
        all_y.append(y)
        all_flow_groups.append(flow_groups)
        
        minute = int(filename.split("/")[-1][:-4])
        all_minutes += [minute for m in range(len(y))]
    
    all_X = pd.concat(all_X).to_dict("records")
    all_y = np.concatenate(all_y)
    all_flow_groups = np.concatenate(all_flow_groups)
    
    return all_X, all_y, all_flow_groups, all_minutes

def get_X_y_binary_quantized(
    filenames: list[str], 
    feature_names: list[str], 
    min_packets: int,
    percentile: float,
    verbose: bool,
    quantizer_path: str,
    q_feat_idx: list[int]
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    
    for f in filenames:
        
        # Load minute by minute
        df = pd.read_csv(f, index_col=None, header=0)
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        cutoff = get_cutoff(sizes, percentile)
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Get binary label
        y = get_binary_labels(sizes, cutoff)
        
        # Append
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    # Quantization
    with open(quantizer_path, "rb") as f:
        quantizer = pickle.load(f) 
    q_feat = X[:, q_feat_idx]
    q_feat = quantizer.transform(q_feat)
    X = np.delete(X, q_feat_idx, axis=1)
    X = np.concatenate((X, q_feat), axis=1)
    
    return X, y 


def get_X_y_regression(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        verbose: bool,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    
    for f in filenames:
        
        # Load minute by minute
        df = pd.read_csv(f, index_col=None, header=0)
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Append
        X_list.append(X)
        y_list.append(sizes)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    return X, y 


def get_X_y_binary_simulate_sampling(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        percentile: float,
        verbose: bool,
        sampling_rate: float=0.02,
        eleph_prop: float=0.3,   
        
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    
    for f in filenames:
        
        # Load minute by minute
        df = pd.read_csv(f, index_col=None, header=0)
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        cutoff = get_cutoff(sizes, percentile)
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Get binary label
        y = get_binary_labels(sizes, cutoff)
        
        # Simulate sampling
        
        n_samples = int(sampling_rate * y.shape[0])
        n_eleph = int(n_samples * eleph_prop)
        n_mice = n_samples - n_eleph
        
        eleph_idx = np.where(y == 1)
        np.random.shuffle(eleph_idx)
        eleph_idx = eleph_idx[:n_eleph]
        
        mice_idx = np.where(y == 0)
        np.random.shuffle(mice_idx)
        mice_idx = mice_idx[:mice_idx]
        
        sampled_idx = np.concatenate([eleph_idx, mice_idx])
        
        # Append sampled instances
        X_list.append(X[sampled_idx])
        y_list.append(y[sampled_idx])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    return X, y 


def get_X_y_binary(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        percentile: float,
        verbose: bool,
        dry_run: int=0,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    
    for f in filenames:
        
        # Load minute by minute
        if dry_run:
            print(f"Dry-run. Capping nrows at {dry_run}")
            df = pd.read_csv(f, index_col=None, header=0, nrows=dry_run)
        else:
            df = pd.read_csv(f, index_col=None, header=0, engine="pyarrow")
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        cutoff = get_cutoff(sizes, percentile)
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Get binary label
        y = get_binary_labels(sizes, cutoff)
        
        # Append
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    return X, y 


def get_X_y_binary_size_cutoffs(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        percentile: float,
        verbose: bool,
        dry_run: int=0, 
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    y_size_list = []
    cutoffs_list = []
    
    for f in filenames:
        
        # Load minute by minute
        if dry_run:
            df = pd.read_csv(f, index_col=None, header=0, nrows=dry_run)
            print(f"Dry-run. Capping at {dry_run} examples.")
        else:
            df = pd.read_csv(f, index_col=None, header=0)
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        cutoff = get_cutoff(sizes, percentile)
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Get binary label
        y = get_binary_labels(sizes, cutoff)
        
        # Append
        X_list.append(X)
        y_list.append(y)
        y_size_list.append(sizes)
        cutoffs_list.append(cutoff)
        
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    sizes = np.concatenate(y_size_list)
    
    return X, y, sizes, cutoffs_list 


def get_X_y_groups_binary(
        filenames: list[str], 
        feature_names: list[str], 
        min_packets: int,
        percentile: float,
        verbose: bool,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
    X_list = []
    y_list = []
    
    for f in filenames:
        
        # Load minute by minute
        df = pd.read_csv(f, index_col=None, header=0)
        X = df[feature_names].to_numpy()
        sizes = df['flow_size'].to_numpy()
        cutoff = get_cutoff(sizes, percentile)
        
        # remove flows below threshold
        X, sizes = remove_min_packet_flows(
            X, 
            sizes, 
            min_packets, 
            verbose
        )
        
        # Get binary label
        y = get_binary_labels(sizes, cutoff)
        
        # Append
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Compute flow IDs based on flow tuple
    flow_tuple_features = [
        i 
        for i, f 
        in enumerate(feature_names) 
        if (("ip" in f) or ("port" in f))
        ]
    if len(flow_tuple_features) > 0:
        _, flow_groups = np.unique(
            X[:, flow_tuple_features],
            axis=0, 
            return_inverse=True
            )
    else:
        flow_groups = np.arange(X.shape[0])
    
    return X, y, flow_groups 


# class MaxXGBNodes(xgb.callback.TrainingCallback):
#     '''
#     Stop training if the model exceeds 1 M nodes

#     '''
    
#     def __init__(self, max_nb_nodes):
#         self.max_nb_nodes = max_nb_nodes

#     def _get_key(self, data, metric):
#         return f'{data}-{metric}'

#     def after_iteration(self, model, epoch, evals_log):
#         '''Update the plot.'''
#         nb_nodes = count_nodes_booster(model)
#         # print("Nb nodes: {}".format(nb_nodes))
#         if nb_nodes < self.max_nb_nodes:
#             # False to indicate training should not stop.
#             return False
#         # True to indicate training should stop because the model is too large
#         return True

    
# def train_model_ga(model_name, param_grid, X_train, y_train, y_bin_train, percentile, logbook_save_path, seed):
#     # Cross validation split
#     if "reg" in model_name:
#         cv = KFold(n_splits=3, shuffle=True, random_state=seed)
#     else:
#         cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
#     # Callbacks
#     logbook_callback = LogbookSaver(checkpoint_path=logbook_save_path)
#     delta_callback = DeltaThreshold(threshold=0.001, generations=5, metric='fitness_min')
#     callbacks = [logbook_callback, delta_callback]

#     # Genetic algorithm

#     if model_name == 'xgb_bin':
#         model = xgb.XGBClassifier(
#             use_label_encoder=False,
#             scale_pos_weight=percentile/(100-percentile), # e.g., 90 / 10 = 9
# #             nthread=90, 
#             eval_metric='logloss',
#             early_stopping_rounds=10,
#             random_state=seed,
# #             tree_method='gpu_hist',
# #             gpu_id=0,
#         )
#         evolved_estimator = GASearchCV(
#             estimator=model,
#             cv=cv,
#             scoring='average_precision',
#             population_size=10,
#             generations=35,
#             tournament_size=3,
#             elitism=True,
#             crossover_probability=0.8,
#             mutation_probability=0.1,
#             param_grid=param_grid,
#             criteria='max',
#             algorithm='eaMuPlusLambda',
#             n_jobs=90,
#             verbose=True,
#             keep_top_k=4
#         )
#         evolved_estimator.fit(X_train, y_bin_train, callbacks=callbacks)

#     if model_name == "xgb_reg":
#         model = xgb.XGBRegressor(
#             use_label_encoder=False,
# #             nthread=90, 
#             eval_metric='mae',
#             early_stopping_rounds=10,
#             random_state=seed,
# #             tree_method='gpu_hist',
# #             gpu_id=0,
#         )
#         evolved_estimator = GASearchCV(
#             estimator=model,
#             cv=cv,
#             scoring='neg_mean_absolute_error',
#             population_size=10,
#             generations=35,
#             tournament_size=3,
#             elitism=True,
#             crossover_probability=0.8,
#             mutation_probability=0.1,
#             param_grid=param_grid,
#             criteria='max',
#             algorithm='eaMuPlusLambda',
#             n_jobs=90,
#             verbose=True,
#             keep_top_k=4
#         )
#         evolved_estimator.fit(X_train, y_train, callbacks=callbacks)
        
#     if model_name == "rf_reg":
#         model = RandomForestRegressor(
#             random_state=seed, 
# #             n_jobs=90, 
#         )
#         evolved_estimator = GASearchCV(
#             estimator=model,
#             cv=cv,
#             scoring='neg_mean_absolute_error',
#             population_size=10,
#             generations=35,
#             tournament_size=3,
#             elitism=True,
#             crossover_probability=0.8,
#             mutation_probability=0.1,
#             param_grid=param_grid,
#             criteria='max',
#             algorithm='eaMuPlusLambda',
#             n_jobs=90,
#             verbose=True,
#             keep_top_k=4
#         )
#         evolved_estimator.fit(X_train, y_train, callbacks=callbacks)
        
#     if model_name == "rf_bin":
#         model = RandomForestClassifier(
#             random_state=seed, 
# #             n_jobs=90, 
#             class_weight={1:percentile, 0:100-percentile }
#         )
#         evolved_estimator = GASearchCV(
#             estimator=model,
#             cv=cv,
#             scoring='average_precision',
#             population_size=10,
#             generations=35,
#             tournament_size=3,
#             elitism=True,
#             crossover_probability=0.8,
#             mutation_probability=0.1,
#             param_grid=param_grid,
#             criteria='max',
#             algorithm='eaMuPlusLambda',
#             n_jobs=90,
#             verbose=True,
#             keep_top_k=4
#         )
#         evolved_estimator.fit(X_train, y_bin_train, callbacks=callbacks)

#     return evolved_estimator
        
    
# def train_model_gs(model_name, c, X_train, y_train, y_bin_train, X_val, y_val, y_bin_val, percentile, max_nb_nodes, seed, n_threads):
    
#     if model_name == "xgb_bin":
#         model = xgb.XGBClassifier(
#             use_label_encoder=False,
#             scale_pos_weight=percentile/(100-percentile), # e.g., 90 / 10 = 9
#             n_estimators= c['num_rounds'],
#             subsample= c['subsample'],
#             reg_alpha= c['reg_alpha'],
#             reg_lambda= c['reg_lambda'],
#             max_depth= c['max_depth'],
#             min_split_loss= c['min_split_loss'],
#             eta= c['eta'],
#             min_child_weight= c['min_child_weight'],
#             nthread=n_threads, 
#             eval_metric='logloss',
#             early_stopping_rounds=10,
#             random_state=seed,
#         )
#         max_nb_nodes = MaxXGBNodes(max_nb_nodes)
#         # model.fit(X_train, y_bin_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(X_val, y_bin_val)], verbose=verbosity)
#         model.fit(X_train, y_bin_train, eval_metric="logloss", eval_set=[(X_val, y_bin_val)], callbacks=[max_nb_nodes], verbose=False)
        

#     if model_name == "xgb_reg":
#         model = xgb.XGBRegressor(
#             use_label_encoder=False,
#             n_estimators= c['num_rounds'],
#             subsample= c['subsample'],
#             reg_alpha= c['reg_alpha'],
#             reg_lambda= c['reg_lambda'],
#             max_depth= c['max_depth'],
#             min_split_loss= c['min_split_loss'],
#             eta= c['eta'],
#             min_child_weight= c['min_child_weight'],
#             nthread=n_threads, 
#             eval_metric='mae',
#             early_stopping_rounds=10,
#             random_state=seed,
#         )
#         max_nb_nodes = MaxXGBNodes(max_nb_nodes)
#         # model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="mae", eval_set=[(X_val, y_val)], verbose=verbosity)
#         model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_val, y_val)], callbacks=[max_nb_nodes], verbose=False)
        
      
#     if model_name == "rf_reg":
#         model = RandomForestRegressor(
#             random_state=seed, 
#             n_jobs=n_threads, 
#             n_estimators=c['n_estimators'] ,
#             max_features=c['max_features'],
#             min_samples_split=c['min_samples_split'],
#             max_depth=c['max_depth'],
#             min_samples_leaf=c['min_samples_leaf'],
#         )
#         model.fit(X_train, y_train)

#     if model_name == "rf_bin":
#         model = RandomForestClassifier(
#             random_state=seed, 
#             n_jobs=n_threads, 
#             n_estimators=c['n_estimators'] ,
#             max_features=c['max_features'],
#             min_samples_split=c['min_samples_split'],
#             max_depth=c['max_depth'],
#             min_samples_leaf=c['min_samples_leaf'],
#             class_weight={1:percentile, 0:100-percentile }
#         )
#         model.fit(X_train, y_bin_train)

#     return model

import numpy as np
import copy
import seaborn as sns
import sklearn
import itertools
import math
from sklearn.ensemble import RandomForestClassifier
# from sklearn_genetic.space import Continuous, Categorical, Integer


def get_predictions_regression(learner, test_set):
    return learner.predict(test_set)
    # return np.exp(learner.predict(test_set))


# def create_params_dicts_rf(params_grid, genetic_algo=False):
#     if genetic_algo:
#         params_dicts = {}
#         for k, v in params_grid.items():
#             if v['type'] == 'Integer':
#                 params_dicts[k] = Integer(v['values'][0], v['values'][1])
#             if v['type'] == 'Continuous':
#                 params_dicts[k] = Continuous(v['values'][0], v['values'][1])
#             if v['type'] == 'Categorical':
#                 params_dicts[k] = Categorical(v['values'])
        
#     else:
#         params_dicts = []
#         combinations = list(itertools.product(*params_grid.values()))
#         for c in combinations:
#             params = {
#                 'n_estimators': c[0],
#                 'max_features': c[1],
#                 'min_samples_split': c[2],
#                 'max_depth': c[3],
#                 'min_samples_leaf': c[4],
#             }
#             params_dicts.append(params)

#     return params_dicts


# def create_params_dicts_xgb(params_grid, genetic_algo=False):
    
#     if genetic_algo:
#         params_dicts = {}
#         for k, v in params_grid.items():
#             if v['type'] == 'Integer':
#                 params_dicts[k] = Integer(v['values'][0], v['values'][1])
#             if v['type'] == 'Continuous':
#                 params_dicts[k] = Continuous(v['values'][0], v['values'][1])
#             if v['type'] == 'Categorical':
#                 params_dicts[k] = Categorical(v['values'])
        
#     else:
#         params_dicts = []
#         combinations = list(itertools.product(*params_grid.values()))
#         for c in combinations:
#             params = {
#                 'num_rounds': c[0],
#                 'subsample': c[1],
#                 'reg_alpha': c[2],
#                 'reg_lambda': c[3],
#                 'max_depth': c[4],
#                 'min_split_loss': c[5],
#                 'eta': c[6],
#                 'min_child_weight': c[7],
#             }
#             params_dicts.append(params)

#     return params_dicts


def get_predictions_bin_classification(learner, test_set): 
    return learner.predict_proba(test_set)[:, 1] 


def get_p_r_ap(true_binary, preds):
    p, r, thresh = sklearn.metrics.precision_recall_curve(true_binary, preds)
    ap = round(sklearn.metrics.average_precision_score(true_binary, preds), 2)
    return [p, r, thresh, ap]


# TODO: use with y > 1
def remove_min_packet_flows(X, y, min_packets, verbose):
    size_before_filtering = y.shape[0]
    legal_idx = np.where(y >= min_packets)[0]
    if X is not None:
        X = X[legal_idx]
    y = y[legal_idx]
    if verbose:
        print("Removed {}% of flows from dataset (min {} packets per flow)".format(str(100.0 - len(legal_idx)/size_before_filtering * 100.0), min_packets))
    return X, y

            
def count_nodes_rf(learner):
    nb_nodes = []
    for t in learner.estimators_:
        nb_nodes.append(t.tree_.node_count)
    nb_total_nodes = sum(nb_nodes)
    return nb_total_nodes


def size_rf(
    learner:RandomForestClassifier, 
    elevenbits_featuresidx:list,
    twobyte_featuresidx:list
) -> int:
    
    # Revisited from 
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    
    nb_split_nodes_binary = 0
    nb_split_nodes_elevenbits = 0
    nb_split_nodes_twobyte = 0
    nb_leaf_nodes = 0
    
    for clf in learner.estimators_:
        
        # Walk the tree
        
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            # If the left and right child of a node is not the same
            # we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth
            # to `stack` so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # Count split and leaf nodes
    
        for i in range(n_nodes):
            if is_leaves[i]:
                nb_leaf_nodes += 1
            else:
                if feature[i] in elevenbits_featuresidx:  # Float features
                    nb_split_nodes_elevenbits += 1
                elif feature[i] in twobyte_featuresidx:  # Float features
                    nb_split_nodes_twobyte += 1
                else:
                    nb_split_nodes_binary += 1
                    
    # Model size in bytes
    size_feature_index = math.ceil(math.log(clf.n_features_in_, 2)) 
    model_size_bytes = (
        (size_feature_index * nb_split_nodes_binary * 1) + 
        (size_feature_index * nb_split_nodes_elevenbits * 11) +
        (size_feature_index * nb_split_nodes_twobyte * 16) +
        (nb_leaf_nodes * 1)
    ) / 8
    
    return int(model_size_bytes)


def count_nodes_xgb(learner):
    nb_nodes = []
    for t in learner.get_booster().get_dump():
        nb_nodes.append(len(t.split('\n')) - 1)
    nb_total_nodes = sum(nb_nodes)
    return nb_total_nodes


def count_max_nodes(depth, n_trees):
    return (2**(depth+1) -1)*n_trees


def count_nodes_booster(booster):
    nb_nodes = []
    for t in booster.get_dump():
        nb_nodes.append(len(t.split('\n')) - 1)
    nb_total_nodes = sum(nb_nodes)
    return nb_total_nodes


def get_cutoff(y, percentile):
    cutoff = np.percentile(y, percentile) 
    return cutoff
    
    
def get_binary_labels(y, cutoff):
    binary_y = copy.deepcopy(y)
    binary_y[y <= cutoff] = 0
    binary_y[y > cutoff] = 1
    return binary_y


def get_name_model(model_name, combination):
    c = copy.deepcopy(combination)
    for k, v in c.items():
        if v == None:
            v = "None"
    if model_name == "xgb_bin":
        save_name = 'bin_xgb_'+str(c['num_rounds'])+'rounds_'+str(c['subsample'])+'subs_'+str(c['reg_alpha'])+'rega_'+str(c['reg_lambda'])+'regl_'+str(c['max_depth'])+'maxd_'+str(c['min_split_loss'])+'minspl_'+str(c['eta'])+'eta_'+str(c['min_child_weight'])+'minchldw'+'.pkl'

    if model_name == "xgb_reg":
        save_name = 'reg_xgb_'+str(c['num_rounds'])+'rounds_'+str(c['subsample'])+'subs_'+str(c['reg_alpha'])+'rega_'+str(c['reg_lambda'])+'regl_'+str(c['max_depth'])+'maxd_'+str(c['min_split_loss'])+'minspl_'+str(c['eta'])+'eta_'+str(c['min_child_weight'])+'minchldw'+'.pkl'
        
    if model_name == "rf_reg":
        save_name = 'reg_rf_'+str(c['n_estimators'])+'trees_'+str(c['max_features'])+'maxf_'+str(c['min_samples_split'])+'minsp_'+str(c['max_depth'])+'maxd_'+str(c['min_samples_leaf'])+'minsa'+'.pkl'
       
    if model_name == "rf_bin":
        save_name = 'bin_rf_'+str(c['n_estimators'])+'trees_'+str(c['max_features'])+'maxf_'+str(c['min_samples_split'])+'minsp_'+str(c['max_depth'])+'maxd_'+str(c['min_samples_leaf'])+'minsa'+'.pkl'
        
    return save_name