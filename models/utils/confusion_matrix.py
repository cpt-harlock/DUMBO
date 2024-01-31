import pickle
import argparse
from statistics import fmean


def main(args):
    all_tpr = []
    all_tnr = []
    all_fpr = []
    all_fnr = []

    with open(args.model_stats_path, "rb") as f:
        model_stats = pickle.load(f)

    if args.pheavy_stage:
        conf_mats = model_stats["initial_model_conf_mats_proba_thr"]
    else:
        conf_mats = model_stats["cl_model_conf_mats"]

    for i, conf_mat in enumerate(conf_mats):
        tp = conf_mat[1][1]
        fn = conf_mat[1][0]
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fnr = fn / (tp + fn)
        fpr = fp / (tn + fp)
        all_tpr.append(tpr)
        all_tnr.append(tnr)
        all_fnr.append(fnr)
        all_fpr.append(fpr)

    mean_tpr = fmean(all_tpr)
    mean_tnr = fmean(all_tnr)
    mean_fpr = fmean(all_fpr)
    mean_fnr = fmean(all_fnr)

    print(f"{mean_tpr=}")
    print(f"{mean_tnr=}")
    print(f"{mean_fpr=}")
    print(f"{mean_fnr=}")

    rates_path = "/".join(args.model_stats_path.split("/")[:-1]) + "/rates.txt"
    with open(rates_path, "w") as f:
        f.write(f"TPR,{mean_tpr}\n")
        f.write(f"TNR,{mean_tnr}\n")
        f.write(f"FPR,{mean_fpr}\n")
        f.write(f"FNR,{mean_fnr}\n")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-stats-path", help="Path to the model statistics in a pickle file", type=str, required=True)
    argparser.add_argument("--pheavy-stage", help="pHeavy stage to consider", type=int, required=False, default=0)
    args = argparser.parse_args()
    
    main(args)