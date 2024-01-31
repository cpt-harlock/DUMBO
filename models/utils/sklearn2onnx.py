import os
import sys
module_path = os.path.abspath(os.path.join('./models/'))
# module_path = os.path.abspath(os.path.join('/home/razorin/sapienza-coda/models/'))
if module_path not in sys.path:
    sys.path.append(module_path)
# module_path = os.path.abspath(os.path.join('/home/razorin/sapienza-coda/models/flow_size_hh/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from helpers import get_X_y_binary
from training.voting_rf_classifier import VotingRandomForestClassifier
from training.training_testing import get_threshold

import argparse
import numpy as np
import random
import pandas as pd
import pickle
from copy import deepcopy
from statistics import mean
import numbers
import onnxruntime as rt

import sklearn
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import to_onnx
from skl2onnx import update_registered_converter
from skl2onnx.shape_calculators.ensemble_shapes import calculate_tree_classifier_output_shapes
from skl2onnx.common.data_types import (
    FloatTensorType, 
    StringTensorType,
)
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.common.utils_classifier import get_label_classes
from skl2onnx.common.data_types import (
    BooleanTensorType, 
    Int64TensorType, 
    guess_numpy_type)
from skl2onnx.common.tree_ensemble import (
    add_tree_to_attribute_pairs_hist_gradient_boosting,
    get_default_tree_classifier_attribute_pairs,
    sklearn_threshold
)
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers import collect_intermediate_steps, compare_objects


SEED = 42
random.seed(SEED)
np.random.seed(SEED)   


def add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id,
             feature_id, mode, value, true_child_id, false_child_id,
             weights, weight_id_bias, leaf_weights_are_counts,
             adjust_threshold_for_sklearn, dtype,
             nodes_missing_value_tracks_true=False):
    attr_pairs['nodes_treeids'].append(tree_id)
    attr_pairs['nodes_nodeids'].append(node_id)
    attr_pairs['nodes_featureids'].append(feature_id)
    attr_pairs['nodes_modes'].append(mode)
    if adjust_threshold_for_sklearn and mode != 'LEAF':
        attr_pairs['nodes_values'].append(
            sklearn_threshold(value, dtype, mode))
    else:
        attr_pairs['nodes_values'].append(value)
    attr_pairs['nodes_truenodeids'].append(true_child_id)
    attr_pairs['nodes_falsenodeids'].append(false_child_id)
    attr_pairs['nodes_missing_value_tracks_true'].append(nodes_missing_value_tracks_true)
    attr_pairs['nodes_hitrates'].append(1.)

    # Add leaf information for making prediction
    if mode == 'LEAF':
        factor = tree_weight 
        flattened_weights = weights.flatten()

        # If the values stored at leaves are counts of possible classes, we
        # need convert them to probabilities by doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            flattened_weights = flattened_weights / s
            
            # Simulate voting
            if flattened_weights[1] < args.proba_thr:
                flattened_weights = [1., 0.]
            else:
                flattened_weights = [0., 1.] 
                
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]
        
        # Note that attribute names for making prediction are different for
        # classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs['class_treeids'].append(tree_id)
                attr_pairs['class_nodeids'].append(node_id)
                attr_pairs['class_ids'].append(i + weight_id_bias)
                attr_pairs['class_weights'].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs['target_treeids'].append(tree_id)
                attr_pairs['target_nodeids'].append(node_id)
                attr_pairs['target_ids'].append(i + weight_id_bias)
                attr_pairs['target_weights'].append(w)

def add_tree_to_attribute_pairs(
    attr_pairs, is_classifier, tree, tree_id,
    tree_weight, weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False, dtype=None
):
    for i in range(tree.node_count):
        node_id = i
        weight = tree.value[i]

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = 'BRANCH_LEQ'
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
        else:
            mode = 'LEAF'
            feat_id = 0
            threshold = 0.
            left_child_id = 0
            right_child_id = 0

        add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id,
                 feat_id, mode, threshold, left_child_id, right_child_id,
                 weight, weight_id_bias, leaf_weights_are_counts,
                 adjust_threshold_for_sklearn=adjust_threshold_for_sklearn,
                 dtype=dtype, 
                )
     
def convert_sklearn_random_forest_classifier_thresh(
    scope, 
    operator, 
    container, 
    op_type='TreeEnsembleClassifier',
    op_domain='ai.onnx.ml', 
    op_version=1,
):
    
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    attr_dtype = dtype if op_version >= 3 else np.float32
    op = operator.raw_operator

    if hasattr(op, 'n_outputs_'):
        n_outputs = int(op.n_outputs_)
        options = container.get_options(
            op, dict(raw_scores=False, decision_path=False,
                     decision_leaf=False))
    elif hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingClassifier
        n_outputs = op.n_trees_per_iteration_
        options = container.get_options(op, dict(raw_scores=False))
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or "
            "'n_trees_per_iteration_'.")

    use_raw_scores = options['raw_scores']

    if n_outputs == 1 or hasattr(op, 'loss_') or hasattr(op, '_loss'):
        classes = get_label_classes(scope, op)

        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)
        attr_pairs = get_default_tree_classifier_attribute_pairs()
        attr_pairs['name'] = scope.get_unique_operator_name(op_type)

        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attr_pairs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, str) for i in classes):
            class_labels = [str(i) for i in classes]
            attr_pairs['classlabels_strings'] = class_labels
        else:
            raise ValueError(
                'Only string and integer class labels are allowed.')

        # random forest calculate the final score by averaging over all trees'
        # outcomes, so all trees' weights are identical.
        if hasattr(op, 'estimators_'):
            estimator_count = len(op.estimators_)
            tree_weight = 1. / estimator_count
        elif hasattr(op, '_predictors'):
            # HistGradientBoostingRegressor
            estimator_count = len(op._predictors)
            tree_weight = 1.
        else:
            raise NotImplementedError(
                "Model should have attribute 'estimators_' or '_predictors'.")

        for tree_id in range(estimator_count):
            # print(tree_id)

            if hasattr(op, 'estimators_'):
                tree = op.estimators_[tree_id].tree_
                add_tree_to_attribute_pairs(
                    attr_pairs, True, tree, tree_id,
                    tree_weight, 0, True, True,
                    dtype=dtype)
            else:
                # HistGradientBoostClassifier
                if len(op._predictors[tree_id]) == 1:
                    tree = op._predictors[tree_id][0]
                    add_tree_to_attribute_pairs_hist_gradient_boosting(
                        attr_pairs, True, tree, tree_id, tree_weight, 0,
                        False, False, dtype=dtype)
                else:
                    for cl, tree in enumerate(op._predictors[tree_id]):
                        add_tree_to_attribute_pairs_hist_gradient_boosting(
                            attr_pairs, True, tree, tree_id * n_outputs + cl,
                            tree_weight, cl, False, False,
                            dtype=dtype)

        if hasattr(op, '_baseline_prediction'):
            if isinstance(op._baseline_prediction, np.ndarray):
                attr_pairs['base_values'] = list(
                    op._baseline_prediction.ravel())
            else:
                attr_pairs['base_values'] = [op._baseline_prediction]

        if hasattr(op, 'loss_'):
            loss = op.loss_
        elif hasattr(op, '_loss'):
            # scikit-learn >= 0.24
            loss = op._loss
        else:
            loss = None
        if loss is not None:
            if use_raw_scores:
                attr_pairs['post_transform'] = "NONE"
            elif loss.__class__.__name__ in (
                    "BinaryCrossEntropy", "HalfBinomialLoss"):
                attr_pairs['post_transform'] = "LOGISTIC"
            elif loss.__class__.__name__ in (
                    "CategoricalCrossEntropy", "HalfMultinomialLoss"):
                attr_pairs['post_transform'] = "SOFTMAX"
            else:
                raise NotImplementedError(
                    "There is no corresponding post_transform for "
                    "'{}'.".format(loss.__class__.__name__))
        elif use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}' and loss '{}'.".format(type(op), loss))

        input_name = operator.input_full_names
        if isinstance(operator.inputs[0].type, BooleanTensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, input_name, cast_input_name,
                       container, to=onnx_proto.TensorProto.FLOAT)
            input_name = cast_input_name

        if dtype is not None:
            for k in attr_pairs:
                if k in ('nodes_values', 'class_weights',
                         'target_weights', 'nodes_hitrates',
                         'base_values'):
                    attr_pairs[k] = np.array(
                        attr_pairs[k], dtype=attr_dtype).ravel()

        container.add_node(
            op_type, input_name,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain=op_domain, op_version=op_version, **attr_pairs)

        if (not options.get('decision_path', False) and
                not options.get('decision_leaf', False)):
            return
        # decision_path
        tree_paths = []
        tree_leaves = []
        for i, tree in enumerate(op.estimators_):

            attrs = get_default_tree_classifier_attribute_pairs()
            attrs['name'] = scope.get_unique_operator_name(
                "%s_%d" % (op_type, i))
            attrs['n_targets'] = int(op.n_outputs_)
            add_tree_to_attribute_pairs(attrs, True, tree.tree_, 0, 1., 0, False, True, dtype=dtype)
            # add_tree_to_attribute_pairs(attrs, True, tree.tree_, 0, 1., 0, True, True, dtype=dtype)

            attrs['n_targets'] = 1
            attrs['post_transform'] = 'NONE'
            attrs['target_ids'] = [0 for _ in attrs['class_ids']]
            attrs['target_weights'] = [
                float(_) for _ in attrs['class_nodeids']]
            attrs['target_nodeids'] = attrs['class_nodeids']
            attrs['target_treeids'] = attrs['class_treeids']
            rem = [k for k in attrs if k.startswith('class')]
            for k in rem:
                del attrs[k]

            if dtype is not None:
                for k in attrs:
                    if k in ('nodes_values', 'class_weights',
                             'target_weights', 'nodes_hitrates',
                             'base_values'):
                        attrs[k] = np.array(attrs[k], dtype=attr_dtype).ravel()

            if options['decision_path']:
                # decision_path
                tree_paths.append(
                    _append_decision_output(
                        input_name, attrs, _build_labels_path, None,
                        scope, operator, container,
                        op_type=op_type, op_domain=op_domain,
                        op_version=op_version, regression=True,
                        overwrite_tree=tree.tree_))
            if options['decision_leaf']:
                # decision_path
                tree_leaves.append(
                    _append_decision_output(
                        input_name, attrs, _build_labels_leaf, None,
                        scope, operator, container,
                        op_type=op_type, op_domain=op_domain,
                        op_version=op_version, regression=True,
                        cast_encode=True))

        # merges everything
        n_out = 2
        if options['decision_path']:
            apply_concat(
                scope, tree_paths, operator.outputs[n_out].full_name,
                container, axis=1,
                operator_name=scope.get_unique_operator_name('concat'))
            n_out += 1

        if options['decision_leaf']:
            # decision_path
            apply_concat(
                scope, tree_leaves, operator.outputs[n_out].full_name,
                container, axis=1,
                operator_name=scope.get_unique_operator_name('concat'))
            n_out += 1

    else:
        if use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op)))
        concatenated_proba_name = scope.get_unique_variable_name(
            'concatenated_proba')
        proba = []
        for est in op.estimators_:
            reshaped_est_proba_name = scope.get_unique_variable_name(
                'reshaped_est_proba')
            est_proba = predict(
                est, scope, operator, container, op_type, op_domain,
                op_version, is_ensemble=True)
            apply_reshape(
                scope, est_proba, reshaped_est_proba_name, container,
                desired_shape=(
                    1, n_outputs, -1, max([len(x) for x in op.classes_])))
            proba.append(reshaped_est_proba_name)
        apply_concat(scope, proba, concatenated_proba_name,
                     container, axis=0)
        if container.target_opset >= 18:
            axis_name = scope.get_unique_variable_name('axis')
            container.add_initializer(
                axis_name, onnx_proto.TensorProto.INT64, [1], [0])
            container.add_node(
                'ReduceMean', [concatenated_proba_name, axis_name],
                operator.outputs[1].full_name,
                name=scope.get_unique_operator_name('ReduceMean'),
                keepdims=0)
        else:
            container.add_node(
                'ReduceMean', concatenated_proba_name,
                operator.outputs[1].full_name,
                name=scope.get_unique_operator_name('ReduceMean'),
                axes=[0], keepdims=0)
        predictions = _calculate_labels(
            scope, container, op, operator.outputs[1].full_name)
        apply_concat(scope, predictions, operator.outputs[0].full_name,
                     container, axis=1)

        if (options.get('decision_path', False) or
                options.get('decision_leaf', False)):
            raise RuntimeError(
                "Decision output for multi-outputs is not implemented yet.")
        
def sanity_check(X_df, X_df_selected_feat, onnx_model, sklearn_model, proba_thr, is_pheavy, quantization=True):
    # Check opset version
    if not is_pheavy:
        assert (onnx_model.opset_import[0].version == 13) or (onnx_model.opset_import[1].version == 13)

    # Check matching preds
    sess_onnx = rt.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )
    if is_pheavy:
        inputs_onnnx = {"X": X_df_selected_feat.values}
    else:
        if quantization:
            inputs_onnnx = {"X": X_df_selected_feat}
        else:
            inputs_onnnx = {
                c: X_df_selected_feat[c].values.reshape((-1, 1))
                for c 
                in X_df_selected_feat.columns
            }
    p_onnx = sess_onnx.run(None, inputs_onnnx)
    # If there is a tie in the votes, we favor the positive class. 
    # Note: this must be implemented in the simulator (rounding too).
    preds_proba_onnx = p_onnx[1][:, 1].round(5)
    if is_pheavy:
        preds_onnx = np.where(preds_proba_onnx <= 0.5, 0, 1)
        preds_sklearn = sklearn_model.predict(X_df)
    else:
        preds_onnx = np.where(preds_proba_onnx < 0.5, 0, 1)
        preds_sklearn = sklearn_model.predict(X_df, thresh=proba_thr)

    assert np.array_equal(preds_onnx, preds_sklearn)
            
def main(args):
    print('The scikit-learn version in use is {}.'.format(sklearn.__version__))

    # Setup ONNX
    if args.pheavy:
        registered_options = {
            'nocl': [True], 
            'zipmap': [False],
            "decision_path": [False],
            "decision_leaf": [False],
        }
    else:
        registered_options = {
            'nocl': [True], 
            'zipmap': [False],
            "raw_scores": [False, True],
            "decision_path": [False],
            "decision_leaf": [False],
        }

    update_registered_converter(
        VotingRandomForestClassifier, 'VotingRandomForestClassifier',
        shape_fct=calculate_tree_classifier_output_shapes, 
        convert_fct=convert_sklearn_random_forest_classifier_thresh,
        options=registered_options
    )

    # Load data
    with open(args.features_path) as f:
        feature_names = [line.rstrip() for line in f]
    X_preproc, y = get_X_y_binary(
        filenames=[args.trace], 
        feature_names=feature_names, 
        min_packets=args.n_pk,
        percentile=args.percentile,
        verbose=False,
        dry_run=0,
    )
    X_df = pd.DataFrame(
        data=X_preproc,
        columns=feature_names
    ).astype(np.float32)

    # Load Scikit-learn model
    for model_path in args.skl_models:
        print(f"\nConverting {model_path}")
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        if args.pheavy:
            pipeline = pipeline[args.n_pk]["tree"]
        print(f"Scikit-learn version used to train the model: {pipeline.__getstate__()['_sklearn_version']}")
        assert pipeline.__getstate__()['_sklearn_version'] == sklearn.__version__
        pipeline_cp = deepcopy(pipeline)

        if args.features_selection:
            # Remove the features selection step from the pipeline
            # because Pandas columns names are lost with ONNX conversion
            # print(pipeline['feature_selection'].feature_names_in_) # For debug
            features_selected = pipeline['feature_selection'].transform(X_df).columns.values
            pipeline_cp.steps = pipeline_cp.steps[1:]
            X_df_selected_feat = X_df[features_selected].astype(np.float32)
        else:
            X_df_selected_feat = X_df

        if args.quantization:
            # bins_edges = pipeline['preprocessing'].transformers_[0][1].bin_edges_
            bins_edges_est = pipeline['preprocessing'].transformers_[0][1]
            pipeline_cp.steps = pipeline_cp.steps[1:]
            X_df_selected_feat = pipeline['preprocessing'].transform(X_df_selected_feat)

        if not args.pheavy:
            # Compute probability threshold
            proba_thresh = get_threshold(
                model=pipeline,
                n_packets=args.n_pk,
                pct_hh=[args.share_predicted_hh],
                minute_paths=[args.trace],
                share_evicted=args.share_evicted,
                features=feature_names,
                dry_run=0,
                eleph_tracker_nentries=[] if args.eleph_tracker_nentries == 0 else [args.eleph_tracker_nentries],
                percentile=args.percentile,
            )[args.eleph_tracker_nentries if args.eleph_tracker_nentries > 0 else args.share_predicted_hh]
            print(f"Proba thresh: {proba_thresh}.")
            args.proba_thr = proba_thresh

        # Convert to ONNX model
        onnx_model = to_onnx(
            pipeline_cp,
            X=X_df_selected_feat.values if args.pheavy else X_df_selected_feat, 
            name="votingRF", 
            target_opset=13, # == ONNX runtime 1.8. cf. https://onnxruntime.ai/docs/reference/compatibility.html
            options={
                'zipmap': False,
                'nocl': True,
                "decision_leaf": False,
                "decision_path": False,
            } if args.pheavy else {
                'zipmap': False,
                'nocl': True,
                "raw_scores": True,
                "decision_leaf": False,
                "decision_path": False,
            }, 
            verbose=0,
            initial_types=None,
        )
        sanity_check(X_df, X_df_selected_feat, onnx_model, pipeline, args.proba_thr, args.pheavy, args.quantization)

        # Save ONNX model
        model_name = model_path.split("/")[-1]
        onnx_path = "/".join(model_path.split("/")[:-1])
        onnx_path = onnx_path + "/" + model_name [:-4] + ".onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        if args.features_selection:
            with open(onnx_path[:-5] + "_selectedfeatures.pkl", "wb") as f:
                pickle.dump(features_selected, f)
        if args.quantization:
            with open(onnx_path[:-5] + "_bins_edges.pkl", "wb") as f:
                pickle.dump(bins_edges_est, f)
        print(f"Saved at {onnx_path}")
                

if __name__ == "__main__":
    
    # example usage on netMl06 (sudo is required to write to /mnt/storage/*): 

    # Voting Random Forest (ours)

    # CAIDA
    # sudo ~/conda_envs/flow2vec/bin/python ./utils/sklearn2onnx.py
    # --skl-models /mnt/storage/raphael/coda/train_val_models/5_pk/first_pk/TCP+UDP/initial5min_pruning+feat_selection+quantization_0dryrun_caida1/cl_pipeline_134500.pkl
    # --features-path ./training/params/feature_names_5pk.txt
    # --features-selection 
    # --share-evicted 0.20
    # --share-evicted 0.33765629103758943
    # --share-evicted 0.29
    # --eleph-tracker-nentries 20000
    # --quantization
    # --trace /mnt/storage/raphael/coda/caida/20160121/preprocessed_5-20pk_tcpudpicmp/135000_tcp_udp.csv 
    # --n-pk 5

    # MAWI
    # sudo ~/conda_envs/flow2vec/bin/python ./utils/sklearn2onnx.py
    # --skl-models /mnt/storage/raphael/coda/train_val_models/5_pk/first_pk/TCP+UDP/initial5min_pruning+feat_selection+quantization_0dryrun_mawi/cl_pipeline_201904091915.pkl
    # --features-path ./training/params/feature_names_5pk.txt
    # --features-selection 
    # --share-evicted 0.15
    # --share-evicted 0.11323176786882069
    # --eleph-tracker-nentries 7500
    # --quantization
    # --trace /mnt/storage/raphael/coda/mawi/20190409/preprocessed_5-20pk_tcpudpicmp/201904091921_tcp_udp.csv
    # --n-pk 5

    # UNI2
    # sudo ~/conda_envs/flow2vec/bin/python utils/sklearn2onnx.py 
    # --skl-models /mnt/storage/raphael/coda/train_val_models/5_pk/first_pk/TCP+UDP/initial5min_pruning+feat_selection+quantization_0dryrun_1minepoch_top5pct_uni2/cl_pipeline_2222.pkl 
    # --eleph-tracker-nentries 100 
    # --share-evicted 0.15 
    # --share-evicted 0.29 
    # --features-path ./training/params/feature_names_5pk.txt 
    # --features-selection 
    # --percentile 99
    # --quantization
    # --trace /mnt/storage/raphael/coda/uni2/1minepoch/20100122/preprocessed_5-20pk_tcpudpicmp/2222_tcp_udp.csv 
    # --n-pk 5

    # pHeavy

    # CAIDA (old)
    # sudo ~/conda_envs/flow2vec/bin/python ./utils/sklearn2onnx.py 
    # --skl-models /mnt/storage/raphael/coda/train_val_models/5_pk/first_pk/TCP+UDP/pheavy_5-20_thr0.6_0dryrun_caida1/model_pheavy.pkl 
    # --proba-thr 0.07 (unused, pHeavy ONNX model return the real class probabilities)
    # --features-path ./training/params/feature_names_pheavy_5pk.txt 
    # --trace /mnt/storage/raphael/coda/caida/20160121/preprocessed_5-20pk_tcpudpicmp/130000_tcp.csv 
    # --n-pk 5 
    # --pheavy
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--skl-models", nargs="+", help="List of paths to Sklearn models", type=str, required=True)
    argparser.add_argument("--proba-thr", nargs="+", help="Cutoff applied on the predicted probability", type=float, default=0.07)
    argparser.add_argument("--features-path", help="Path to a txt file containing the list of features used as input by the model", type=str, required=True)
    argparser.add_argument("--features-selection", help="If the model performs feature selection as a first step", action="store_true", default=False)
    argparser.add_argument("--quantization", help="If the model performs quantization as a second step", action="store_true", default=False)
    argparser.add_argument("--trace", help="Sample trace used to infer feature types", type=str, required=True)
    argparser.add_argument("--n-pk", help="Model used for the k-th packet arrival", type=int, required=True)
    argparser.add_argument("--pheavy", help="If the model to convert is a pHeavy tree", action="store_true", default=False)
    argparser.add_argument("--percentile", help="Elephants definition (n-th precentile)", type=float, default=99)
    argparser.add_argument("--eleph-tracker-nentries", help="Number of desired predicted elephants", type=int, required=True)
    argparser.add_argument("--share-predicted-hh", help="Share of desired predicted elephants (unused if --eleph-tracker-nentries > 0)", type=float, required=False, default=0.)
    argparser.add_argument("--share-evicted", help="Fraction of flows with > k pk that are evicted too early from the Flow Manager (collisions)", type=float, required=True)
    args = argparser.parse_args()
    
    main(args)