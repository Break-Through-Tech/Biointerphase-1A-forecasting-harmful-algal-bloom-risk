"""Scores for the three-class weekly forecast.

Macro-F1 is the main score, so the rare at-risk and bloom weeks count as much as
no-bloom weeks. Accuracy and Cohen's kappa are also reported. Medina et al. (2024,
§2.4) note that a forecast lagging behind reality can score well on both yet "miss the
onset of K. brevis blooms"; bloom onsets and escalations measure that.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score)

from .config import CLASS_NAMES

LABELS = [0, 1, 2]


def find_bloom_onsets(y_true: pd.Series, quiet_weeks: int = 4) -> list[pd.Timestamp]:
    """Find the weeks where a bloom starts after a quiet spell.

    A bloom onset is a week observed as bloom (class 2) with no bloom observed in the
    `quiet_weeks` weeks before it.

    Args:
        y_true (pandas.Series): Observed class per week, indexed by week-ending date.
        quiet_weeks (int): Bloom-free weeks needed before a bloom counts as a new onset.

    Returns:
        list[pandas.Timestamp]: The onset weeks, in date order.

    Example:
        >>> y = pd.Series([0, 2, 2, 0, 0, 0, 0, 2],
        ...               index=pd.date_range("2020-01-05", periods=8, freq="W-SUN"))
        >>> [d.date().isoformat() for d in find_bloom_onsets(y)]
        ['2020-01-12', '2020-02-23']
    """
    onsets, last_bloom = [], None
    for week, cls in y_true.items():
        if cls == 2:
            if last_bloom is None or (week - last_bloom).days >= 7 * (quiet_weeks + 1):
                onsets.append(week)
            last_bloom = week
    return onsets


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


def score_onsets(predictions: pd.DataFrame) -> dict:
    """Count how many bloom onsets got a bloom forecast or any warning.

    Args:
        predictions (pandas.DataFrame): Out-of-fold predictions with y_true and y_pred,
            indexed by week (output of evaluation.cross_validate).

    Returns:
        dict: n_onsets, onsets_caught_bloom (forecast bloom) and
        onsets_caught_any_warning (forecast at risk or bloom).

    Example:
        >>> p = pd.DataFrame({"y_true": [0, 2], "y_pred": [0, 1]},
        ...                  index=pd.date_range("2020-01-05", periods=2, freq="W-SUN"))
        >>> score_onsets(p)
        {'n_onsets': 1, 'onsets_caught_bloom': 0, 'onsets_caught_any_warning': 1}
    """
    onsets = find_bloom_onsets(predictions["y_true"])
    p = predictions.loc[onsets, "y_pred"]
    return dict(n_onsets=len(onsets), onsets_caught_bloom=int((p == 2).sum()),
                onsets_caught_any_warning=int((p >= 1).sum()))


def summarise_predictions(predictions: pd.DataFrame) -> dict:
    """Score out-of-fold predictions, including onsets and escalations.

    An escalation is a quiet week (last observed class = no bloom) whose label is at
    risk or bloom. Persistence always says "still quiet" there, so escalation recall
    shows what a model adds beyond repeating last week.

    Args:
        predictions (pandas.DataFrame): Output of evaluation.cross_validate: y_true,
            y_pred, p_no_bloom, p_at_risk, p_bloom and last_class, indexed by week.

    Returns:
        dict: Everything from compute_scores and score_onsets, plus
        agrees_with_persistence (share of weeks forecast as the last observed class),
        n_quiet_weeks, n_escalations, escalation_recall and escalation_precision.

    Example:
        >>> p = pd.DataFrame({"y_true": [0, 1, 2], "y_pred": [0, 1, 2], "last_class": [0, 0, 2],
        ...                   "p_no_bloom": [.8, .2, .1], "p_at_risk": [.1, .7, .2], "p_bloom": [.1, .1, .7]},
        ...                  index=pd.date_range("2020-01-05", periods=3, freq="W-SUN"))
        >>> s = summarise_predictions(p)
        >>> s["n_escalations"], s["escalation_recall"], round(s["agrees_with_persistence"], 2)
        (1, 1.0, 0.67)
    """
    proba = predictions[["p_no_bloom", "p_at_risk", "p_bloom"]].to_numpy()
    m = compute_scores(predictions["y_true"], predictions["y_pred"], proba)
    m.update(score_onsets(predictions))
    m["agrees_with_persistence"] = float((predictions["y_pred"] == predictions["last_class"]).mean())
    quiet = predictions[predictions["last_class"] == 0]
    esc_true, esc_pred = quiet["y_true"] >= 1, quiet["y_pred"] >= 1
    m["n_quiet_weeks"] = int(len(quiet))
    m["n_escalations"] = int(esc_true.sum())
    m["escalation_recall"] = float(recall_score(esc_true, esc_pred, zero_division=0))
    m["escalation_precision"] = float(precision_score(esc_true, esc_pred, zero_division=0))
    return m
