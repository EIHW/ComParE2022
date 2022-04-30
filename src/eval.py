#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from psds_eval import PSDSEval, plot_per_class_psd_roc, plot_psd_roc

_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def do_eval(predictions_dir: str, ground_truth_dir: str, **kwargs) -> None:
    dtc_threshold = 0.5
    gtc_threshold = 0.5
    beta = 1  # Emphasis on FP/TP -> beta: coefficient used to put more (beta > 1) or less (beta < 1) emphasis on false negatives.
    cttc_threshold = 0.0
    alpha_ct = 0.0
    alpha_st = 0.0
    # max_efpr = 170

    # Load metadata and ground truth tables
    ground_truth_csv_path = os.path.join(_ROOT_PATH, ground_truth_dir, "gt.csv")
    metadata_csv_path = os.path.join(_ROOT_PATH, ground_truth_dir, "meta.csv")
    gt_table = pd.read_csv(ground_truth_csv_path, sep="\t")
    meta_table = pd.read_csv(metadata_csv_path, sep="\t")

    pred_dir = os.path.join(_ROOT_PATH, predictions_dir)

    # Instantiate PSDSEval
    psds_eval = PSDSEval(
        dtc_threshold,
        gtc_threshold,
        cttc_threshold,
        ground_truth=gt_table,
        metadata=meta_table,
        class_names=["mosquito"],
    )

    # Add the operating points, with the attached information
    for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
        csv_file = os.path.join(pred_dir, f"baseline_{th:.1f}.csv")
        det_t = pd.read_csv(os.path.join(csv_file), sep="\t")
        info = {"name": f"Op {i + 1}", "threshold": th}
        psds_eval.add_operating_point(det_t, info=info)
        # print(f"\rOperating point {i + 1} added", end=" ")

        # calculate Macro f-score:

        macro_f, class_f = psds_eval.compute_macro_f_score(det_t, beta=beta)
        print(f"\nmacro F-score: {macro_f * 100:.2f}")

    # Calculate the PSD-Score
    psds = psds_eval.psds(alpha_ct, alpha_st)
    print(f"\nPSD-Score: {psds.value:.5f}")

    # Plot the PSD-ROC
    if "filename" in kwargs.keys():
        plot_psd_roc(psds, filename = kwargs["filename"])
    else:
        plot_psd_roc(psds)

    # Plot per class tpr vs fpr/efpr/ctr
    tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
    plot_per_class_psd_roc(
        tpr_vs_fpr,
        psds_eval.class_names,
        title="Per-class TPR-vs-FPR PSDROC",
        xlabel="FPR",
    )
    print((tpr_vs_fpr[1]))
    print((tpr_vs_fpr[0].squeeze()))
    roc_auc = auc(tpr_vs_fpr[1], tpr_vs_fpr[0].squeeze())
    roc_auc_norm = roc_auc / np.max(tpr_vs_fpr[1])
    print("ROC", roc_auc)
    print("ROC norm", roc_auc_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This function evaluates the predictions of the model."""
    )
    parser.add_argument(
        "--predictions_dir",
        default="data/example_predictions/BNN_range",
        help="Folder containing the prediction outputs (baseline_*.csv)",
        type=str,
    )
    parser.add_argument(
        "--ground_truth_dir",
        default="data/ground_truth",
        help="Folder containing the ground truth files to evaluate against (gt.csv, meta.csv)",
        type=str,
    )

    args = parser.parse_args()

    do_eval(args.predictions_dir, args.ground_truth_dir)
