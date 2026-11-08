"""Scores for the three-class weekly forecast.

Macro-F1 is the main score, so the rare at-risk and bloom weeks count as much as
no-bloom weeks. Accuracy and Cohen's kappa are also reported.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score)

from .config import CLASS_NAMES

LABELS = [0, 1, 2]


def compute_scores(y_true: pd.Series, y_pred: pd.Series, proba: np.ndarray | None = None) -> dict:
    """Score a set of forecasts.

    Args:
        y_true (pandas.Series): Observed classes (0, 1, 2).
        y_pred (pandas.Series): Forecast classes.
        proba (numpy.ndarray | None): Optional (n, 3) class probabilities; adds
            ROC-AUC and bloom PR-AUC when given without NaN.

    Returns:
        dict: n, accuracy, balanced_accuracy, macro_f1, kappa, recall and precision per
        class, warning_recall and warning_precision (at risk or bloom counted as one
        "warning" class), and roc_auc_ovr and pr_auc_bloom if `proba` is given.

    Example:
        >>> s = compute_scores(pd.Series([0, 1, 2, 2]), pd.Series([0, 1, 2, 1]))
        >>> [round(float(s[k]), 2) for k in ("accuracy", "recall_bloom", "warning_recall")]
        [0.75, 0.5, 1.0]
    """
    m = dict(
        n=int(len(y_true)),
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred, labels=LABELS),
    )
    rec = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        m[f"recall_{name}"] = rec[i]
        m[f"precision_{name}"] = prec[i]
    warn_true, warn_pred = np.asarray(y_true) >= 1, np.asarray(y_pred) >= 1
    m["warning_recall"] = recall_score(warn_true, warn_pred, zero_division=0)
    m["warning_precision"] = precision_score(warn_true, warn_pred, zero_division=0)
    if proba is not None and not np.isnan(proba).any():
        m["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=LABELS)
        m["pr_auc_bloom"] = average_precision_score(np.asarray(y_true) == 2, proba[:, 2])
    return m
