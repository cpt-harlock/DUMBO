# Example command:
# python model_sizing.py --model /home/razorin/sapienza-coda/models/flow_size_hh/best_models/top_1pct/robust_recent_5tuple/1_pk/retrain_full30min/5TUPLERECENT__ip+port__criterionentropy_max_depth28_max_features0.3046137691733707_min_samples_leaf5_min_samples_split3_n_estimators88.pkl --n-features-binary 96 

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import argparse
import pickle
import numpy as np
from copy import deepcopy
import math

# from helpers import size_rf

import random
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
# from helpers import check_input, get_X_y_binary, get_X_y_binary_quantized
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, KFold
from pathlib import Path
from typing import Union
import numpy.typing as npt
# from model_sizing import size, get_ranges, get_tree, get_subforest
import pickle
from sklearn.metrics import average_precision_score
from copy import deepcopy


def size_xgb(depths, node_counts, f_b, n_b_feat_idx, w):
    """Compute model size in bits, following various implementations"""
    
    # Count the non-binary features
    f_n_b = len(n_b_feat_idx)
    
    # Full Tree implementation
    full_tree_size = sum([
        (2**(d) - 1) * (np.ceil(np.log2(f_b+f_n_b)) + w + 1)
        for d 
        in depths
    ])
    
    # CSR implementation
    csr_size = sum([
        (1 + w + (np.ceil(np.log2(f_b+f_n_b)))) * n \
        + (d - 1) * n \
        + (d * np.ceil(np.log2(n)))
        for d, n 
        in zip(depths, node_counts)
    ])
    
    return full_tree_size, csr_size


def get_ranges(model, features_idx):
    """
        Compute the ranges used to split the provided features
        Code from: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    """
    ranges = []
    nleaves = []
    
    for clf in model.estimators_:

        nleaves_clf = 0
        
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        for i in range(n_nodes):
            if is_leaves[i]:
                nleaves_clf = nleaves_clf + 1
            else:
                if feature[i] in features_idx:
                    ranges.append(threshold[i])

        nleaves.append(nleaves_clf)
                
    return set(sorted(ranges)), nleaves
    

def get_tree(model, tree_idx):
    tree = deepcopy(model)
    tree.estimators_ = [tree.estimators_[tree_idx]]
    return tree


def get_tree_estimator(model, tree_idx):
    copy_model = deepcopy(model)
    return copy_model.estimators_[tree_idx]


def get_subforest(model, tree_idx_list):
    subforest = deepcopy(model)
    subforest.estimators_ = [subforest.estimators_[i] for i in tree_idx_list]
    return subforest
    
    
def get_subforest_estimators(model, tree_idx_list):
    model_copy = deepcopy(model)
    return [model_copy.estimators_[i] for i in tree_idx_list]
    
    
def get_max_nodes_any_layer_tree(forest):
    """ Identify the maximum number of nodes at any given level, any tree."""
    
    last_full_layers = []
    max_node_counts = []
    
    for clf in forest.estimators_:
        # Parse the tree structure
        # Taken from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # Get the number of nodes per level and the max nb of nodes
        unique, counts = np.unique(node_depth, return_counts=True)
        max_node_counts.append(counts.max())
            
    max_node_counts = max(max_node_counts)

    return max_node_counts
    

def size_fpga_dense_binary(model, n_binary_features):
    size = sum([
        (2**(t.get_depth()) - 1) * (np.ceil(np.log2(n_binary_features)) + 1)
        for t 
        in model.estimators_
    ])

    return size


def size_fpga_hybrid_binary(model, n_binary_features):

    # Find the maximum number of nodes in any layer of any tree
    def find_n(clf):
        "Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py"
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        
        n_nodes_by_depth = {}
        for i in range(n_nodes):
            depth = node_depth[i]
            if depth in n_nodes_by_depth:
                n_nodes_by_depth[depth] = n_nodes_by_depth[depth] + 1
            else:
                n_nodes_by_depth[depth] = 1

        return max(list(n_nodes_by_depth.values()))


    n = max([
        find_n(dt)
        for dt
        in model.estimators_
    ])

    m = np.floor(np.log2(n)) + 1

    size = sum([
        (2**(m) - 1) * (np.ceil(np.log2(n_binary_features)) + 1) + n * (t.get_depth() - m) * (np.ceil(np.log2(n_binary_features)) + 1 + np.ceil(n))
        for t 
        in model.estimators_
    ])

    return size


def size_mat_binary(model, n_binary_features):
    n_leaves = [
        t.get_n_leaves()
        for t
        in model.estimators_
    ]
    size = sum([
        l * (2 * n_binary_features + 1)
        for l
        in n_leaves
    ])

    return size

    
def size(model, f_b, n_b_feat_idx, w=2*8, full_forest=None):
    """
    Compute model size in bits, following various implementations
    TODO: check formulas when w > 0 (i.e., non binary features)
    """
    
    # Count the non-binary features
    f_n_b = len(n_b_feat_idx)
    
    # Full Tree implementation
    full_tree_size = sum([
        (2**(t.get_depth() + 1) - 1) * (np.ceil(np.log2(f_b+f_n_b)) + w + 1)
        for t 
        in model.estimators_
    ])

    # CSR implementation - deprecated
    # csr_size = sum([
    #     (1 + w + (np.ceil(np.log2(f_b+f_n_b)))) * t.tree_.node_count \
    #     + ((t.get_depth() + 1) - 1) * t.tree_.node_count \
    #     + ((t.get_depth() + 1) * np.ceil(np.log2(t.tree_.node_count)))
    #     for t 
    #     in model.estimators_
    # ])
    
    # Match Action Table implementation
    ranges, nleaves = get_ranges(model, features_idx=n_b_feat_idx)
    if len(ranges) > 0:
        nbf = np.ceil(np.log2(len(ranges) + 1))
    else:
        nbf = 0
    nbr = len(ranges) + 1 # FIXME: Overly optimistic (i.e., considering a range-able MAT)
    # mat_size = sum([
    #     (2 ** (t.get_depth() - 1)) * (f_b + f_n_b * nbf + 1) \
    #     + f_n_b * (nbr * (w + nbf))
    #     for t 
    #     in model.estimators_
    # ])
    mat_size = sum([
        nleaves_t * (f_b + f_n_b * nbf + 1) \
        + f_n_b * (nbr * (w + nbf))
        for nleaves_t 
        in nleaves
    ])
    
    # Optimized FullTree representation (hybrid)
    N = get_max_nodes_any_layer_tree(full_forest) # Across all layers and all trees
    M = np.floor(np.log2(N)) + 1
    opt_ft_size = sum([
        (2**M - 1) * (np.ceil(np.log2(f_b+f_n_b)) + w + 1) \
        + N * ((t.get_depth() + 1) - M) * (np.ceil(np.log2(f_b+f_n_b)) + w + 1)
        for t 
        in model.estimators_
    ])
  
    return {
        "FullTree": full_tree_size, 
        # "CSR": csr_size, 
        "MAT": mat_size, 
        "Hybrid": opt_ft_size,
    }

    
# if __name__ == "__main__":
    
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument("-m", "--model", help="Model path", type=str, required=True)
#     argparser.add_argument("-fb", "--n-features-binary", help="Number of binary features", type=int, required=True)
#     argparser.add_argument("-ifnb", "--idx-features-non-binary", nargs="+", help="Indexes on the non-binary features", type=int, required=False, default=[])
#     args = argparser.parse_args()
    
#     with open(args.model, "rb") as f:
#         model = pickle.load(f)
        
#     full_tree_size, csr_size, mat_size = size(
#         model=model, 
#         f_b=args.n_features_binary, 
#         n_b_feat_idx=args.idx_features_non_binary, 
#         w=2*8
#     )
    
#     print(f"Model {args.model} sizes in Bytes: ")
#     print(f"- FullTree implementation: {np.ceil(full_tree_size / 8) / 1_000} KB")
#     print(f"- CSR implementation: {np.ceil(csr_size / 8) / 1_000} KB")
#     print(f"- MAT implementation: {np.ceil(mat_size / 8) / 1_000} KB")
    
# import os
# import sys
# module_path = os.path.abspath(os.path.join('/home/razorin/sapienza-coda/models/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# module_path = os.path.abspath(os.path.join('/home/razorin/sapienza-coda/models/flow_size_hh'))
# if module_path not in sys.path:
#     sys.path.append(module_path


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_best_trees_idx(trees_dict, n):
    """Return the ID of the n best trees in the forest"""
    best_trees_idx = []
    trees = deepcopy(trees_dict)
    trees_scores = sorted([t["score"] for t in trees], reverse=True)
    best_n_scores = trees_scores[:n]
    
    while len(best_trees_idx) < n:
        for i, t in enumerate(trees):
            if t["score"] in best_n_scores:
                best_trees_idx.append(i)
    
    return best_trees_idx


def get_least_degrading_trees_idx(trees_list, n):
    pass


# if __name__ == "__main__":
    
#     argparser = argparse.ArgumentParser()
   
#     argparser.add_argument("--forest", help="The path to the model from which to extract subforests", type=str, required=True)
#     argparser.add_argument("--trees", help="Score and size the individual trees", action="store_true", default=False)
#     argparser.add_argument("--max-subforest", help="The maximum number of trees to consider in a subforest", type=int, required=False)
#     argparser.add_argument("--test", nargs="+", help="Preprocessed testing traces CSV", type=str, required=True)
#     argparser.add_argument("--features", nargs="+", help="Feature names", type=str, required=True)
#     argparser.add_argument("--save-folder", help="Saving CSV for random search results", type=str, required=True)
#     argparser.add_argument("--percentile", help="Bottom cutoff for HH definition, in percentile.", type=int, required=False, default=99)
#     argparser.add_argument("--min-packets", help="Minimum number of packets per flow (pre-filtering)", type=int, required=True)
#     argparser.add_argument("-fb", "--n-features-binary", help="Number of binary features", type=int, required=True)
#     argparser.add_argument("-ifnb", "--idx-features-non-binary", nargs="+", help="Indexes of the non-binary features", type=int, required=False, default=[])
#     argparser.add_argument("-q", "--quantizer", help="Path to the Sklearn saved quantizer", type=str, required=False, default=None)
#     argparser.add_argument("--zero-variance-cols", help="VarianceThreshold saved transformer path", type=str, required=False, default=None)
#     argparser.add_argument("-w", "--width", help="Max size in bits to represent a feature", type=int, required=False, default=2*8)
#     argparser.add_argument("--idx-quantized-feat", nargs="+", help="Indices of the features on which the quantizer should be applied", type=int, required=False)
#     argparser.add_argument("--inv-transform-q", help="Use inverse_transform from the quantized data", action="store_true", default=False)

#     args = argparser.parse_args()
    
#     # I/O checks
#     check_input(args.test)

#     # Save args
#     with open(args.save_folder + "args_test_subforest_and_size.json", 'w') as f:
#         json.dump(args.__dict__, f, indent=2)
        
#     # Load the model
#     with open(args.forest, "rb") as f:
#         forest = pickle.load(f)

#     # Load testing data
#     X, y = get_X_y_binary(
#         filenames=args.test, 
#         feature_names=args.features, 
#         min_packets=args.min_packets,
#         percentile=args.percentile,
#         verbose=True
#     )
    
#     if args.quantizer:
#         with open(args.quantizer, "rb") as f_quantizer:
#             quantizer = pickle.load(f_quantizer) 
            
#         X_q_feat = X[:, args.idx_quantized_feat]
#         X_q_feat = quantizer.transform(X_q_feat)
#         X = np.delete(X, args.idx_quantized_feat, axis=1)
#         args.n_features_binary = X.shape[1] + X_q_feat.shape[1]
        
#         if args.inv_transform_q:
#             X_q_feat_inv_transf = quantizer.inverse_transform(X_q_feat)
#             X = np.concatenate((X, X_q_feat_inv_transf), axis=1)
            
#             # TMP for iat_std_2 that is always equal to 0.0
#             # X[:, -1] = 0.0
            
#             # TMP for 99-features models with a 101-features quantizer (removes `protocol` and `iat_std_2`)
#             # X = np.delete(X, [96, 100], axis=1)
            
#         else:
#             X = np.concatenate((X, X_q_feat), axis=1)
            
#         args.idx_features_non_binary = []
#         print(X.shape)    
        
#     if args.zero_variance_cols:
#         with open(args.zero_variance_cols, "rb") as f_remover:
#             zero_variance_cols = pickle.load(f_remover) 
#         X = X[:, ~zero_variance_cols]
#         if args.quantizer:
#             args.n_features_binary = X.shape[1]
#         print(X.shape)    

#     # Get full forest size and score
#     forest_size = min(size(
#         forest,  
#         f_b=args.n_features_binary, 
#         n_b_feat_idx=args.idx_features_non_binary,
#         w=args.width,
#     ))
#     forest_size = np.ceil(forest_size / 8) / 1_000
#     forest_score = average_precision_score(y, forest.predict_proba(X)[:, 1])
    
#     # Save the forest metrics
#     with open(args.save_folder + "forest_" + args.forest.split('/')[-1][:-4] + ".json", 'w') as f:
#         json.dump({"size": forest_size, "score": forest_score}, f)
#     print(f"- Full forest. {forest_size}, {forest_score}\n")
    
#     # Test and size each tree composing the forest
#     if args.trees or args.max_subforest:
        
#         print(f"- Trees composing the full forest.")
#         trees = []
        
#         for i in range(len(forest.estimators_)):
            
#             tree = get_tree(forest, i)
            
#             tree_size = min(size(
#                 tree,  
#                 f_b=args.n_features_binary, 
#                 n_b_feat_idx=args.idx_features_non_binary,
#                 w=args.width,
#             ))
#             tree_size = np.ceil(tree_size / 8) / 1_000
#             tree_score = average_precision_score(y, tree.predict_proba(X)[:, 1])
            
#             trees.append({
#                 "tree": str(i),
#                 "size": tree_size,
#                 "score": tree_score,
#             })
#             print(trees[i])

#         # Save the trees metrics
#         with open(args.save_folder + "trees_" + args.forest.split('/')[-1][:-4] + ".json", 'w') as f:
#             json.dump(trees, f)
            
#     # Test and size the best subforests
#     if args.max_subforest:
        
#         print("\n- Subforests composing the full forest")
#         subforests = []
        
#         for n in range(1, args.max_subforest):
            
#             best_trees_idx = get_best_trees_idx(trees, n)
            
#             subforest_n = get_subforest(forest, best_trees_idx)
            
#             subforest_n_size = min(size(
#                 subforest_n,  
#                 f_b=args.n_features_binary, 
#                 n_b_feat_idx=args.idx_features_non_binary,
#                 w=args.width,
#             ))
#             subforest_n_size = np.ceil(subforest_n_size / 8) / 1_000
            
#             subforest_n_score = average_precision_score(y, subforest_n.predict_proba(X)[:, 1])
            
#             subforests.append({
#                 "n": str(n),
#                 "size": subforest_n_size,
#                 "score": subforest_n_score,
#             })
#             print(subforests[n-1])

#         # Save the trees metrics
#         with open(args.save_folder + "subforests_" + args.forest.split('/')[-1][:-4] + ".json", 'w') as f:
#             json.dump(subforests, f)
        
    
    