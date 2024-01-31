from sklearn.ensemble import RandomForestClassifier
import numpy as np
from statistics import fmean
from skl2onnx.common.data_types import (
    BooleanTensorType, Int64TensorType, guess_numpy_type)
from skl2onnx.common.utils_classifier import get_label_classes
from skl2onnx.common.tree_ensemble import (
    add_tree_to_attribute_pairs_hist_gradient_boosting,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
    sklearn_threshold
)
import numbers
import pandas as pd

class VotingRandomForestClassifier(RandomForestClassifier):
    """Simulates using a voting average to output class prediction 
    (with probability threshold)"""
    
    def predict(self, X, thresh: float):
        
        # Get individual tree class predictions using thresh
        all_trees_preds = []
        for i, tree in enumerate(self.estimators_):
            if isinstance(X, pd.DataFrame):
                tree_p = tree.predict_proba(X.values)
            else:
                tree_p = tree.predict_proba(X)
            if tree_p.shape[1] == 2: 
                tree_p = tree_p[:, 1]
            else:
                # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
                tree_p = tree_p.flatten()
            tree_p = np.where(tree_p < thresh, 0., 1.)
            all_trees_preds.append(tree_p)
        
        preds = np.array(all_trees_preds).mean(axis=0) # Shape: (n_trees, n_samples)
        preds = np.where(preds < 0.5, 0, 1) # If there is a tie, we favor the positive class

        return np.array(preds)   

    def predict_proba_trees(self, X):
        """
        Return the predicted probability of all trees.
        Return shape: (ntrees, nsamples)
        """
        
        # Get individual tree proba predictions
        all_trees_preds_proba = []
        for i, tree in enumerate(self.estimators_):
            if isinstance(X, pd.DataFrame):
                tree_p = tree.predict_proba(X.values)
            else:
                tree_p = tree.predict_proba(X)
            if tree_p.shape[1] == 2: 
                tree_p = tree_p[:, 1]
            else:
                # Cornercase: when a tree is composed of a single root node, predict_proba output shape is (nsamples, 1)
                tree_p = tree_p.flatten()
            all_trees_preds_proba.append(tree_p)
        
        return np.array(all_trees_preds_proba)