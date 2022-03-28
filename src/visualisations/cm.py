#!/usr/bin/env python
# https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
from os import makedirs
from posixpath import dirname, join, relpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from glob import glob

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            # elif c == 0:
            #     annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="bone_r", annot=annot, fmt='', ax=ax, vmax=100, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png")

if __name__=="__main__":
    result_dir = f"./results"
    predictions = glob(f"{result_dir}/**/predictions*.csv", recursive=True)
    cm_dir = f"./visualisations/cms/"
    for p in predictions:
        pred_df = pd.read_csv(p)
        y = pred_df.true.values
        preds = pred_df.prediction.values
        _cm_dir = join(cm_dir, dirname(relpath(p, result_dir)))
        makedirs(_cm_dir, exist_ok=True)
        if "devel" in p:
            partition = "_devel"
        elif "test" in p:
            partition = "_test"
        else:
            partition = ""
        if len(set(pred_df.true.values)) > 1:
            cm_analysis(y, preds, join(_cm_dir, f"cm{partition}"), sorted(set(y)))