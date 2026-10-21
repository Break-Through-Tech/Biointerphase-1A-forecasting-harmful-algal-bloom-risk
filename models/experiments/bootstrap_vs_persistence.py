"""Is a model's gain over persistence bigger than year-to-year noise?

Paired block bootstrap: resample whole bloom years (with replacement) from the
forward-chaining predictions, recompute macro-F1 and warning recall for the model and
for persistence on the same resample, and record the difference. Resampling years,
not weeks, keeps the week-to-week dependence inside a season intact.

Run from the models/ folder, after the models: python experiments/bootstrap_vs_persistence.py
Writes results/h<H>w/experiments/bootstrap_vs_persistence.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

from common.config import RESULTS_DIR  # noqa: E402

N_BOOT = 2000
CANDIDATES = [(m, fs) for m in ["hybrid_lr", "hybrid_xgb", "logistic_regression", "xgboost", "random_forest"]
              for fs in ["kb_only", "long", "full"]]


def load_predictions(model: str, feature_set: str, scheme: str = "forward") -> pd.DataFrame:
    """Read a model's saved out-of-fold predictions.

    Args:
        model (str): Model folder in results/, e.g. "hybrid_lr".
        feature_set (str): Feature set, e.g. "kb_only".
        scheme (str): Validation scheme.

    Returns:
        pandas.DataFrame: The predictions, indexed by week.
    """
    return pd.read_csv(RESULTS_DIR / model / f"predictions__{feature_set}__{scheme}.csv",
                       index_col=0, parse_dates=True)


def compute_metrics(predictions: pd.DataFrame) -> dict:
    """Return the macro-F1 and warning recall of a set of predictions.

    Args:
        predictions (pandas.DataFrame): y_true and y_pred columns.

    Returns:
        dict: macro_f1 and warning_recall.

    Example:
        >>> compute_metrics(pd.DataFrame({"y_true": [0, 1, 2], "y_pred": [0, 2, 2]}))["warning_recall"]
        1.0
    """
    return dict(macro_f1=f1_score(predictions.y_true, predictions.y_pred, labels=[0, 1, 2],
                                  average="macro", zero_division=0),
                warning_recall=recall_score(predictions.y_true >= 1, predictions.y_pred >= 1, zero_division=0))


def bootstrap_difference(joined: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Bootstrap the model-minus-persistence difference by resampling bloom years.

    Args:
        joined (pandas.DataFrame): Model predictions (y_true, y_pred, bloom_year) with
            persistence's forecast in a y_pers column.
        rng (numpy.random.Generator): Random number generator.
        n_boot (int): Number of resamples.

    Returns:
        dict: d_macro_f1 (point difference), d_macro_f1_ci (95% interval as text),
        p_better_macro_f1 (share of resamples where the model is ahead), and the same
        for warning recall (d_warning_recall, p_better_warning_recall).
    """
    years = joined["bloom_year"].unique()
    groups = {y: joined[joined["bloom_year"] == y] for y in years}
    diffs = {"macro_f1": [], "warning_recall": []}
    for _ in range(n_boot):
        sample = pd.concat([groups[y] for y in rng.choice(years, len(years), replace=True)])
        a, b = compute_metrics(sample), compute_metrics(sample.assign(y_pred=sample["y_pers"]))
        for k in diffs:
            diffs[k].append(a[k] - b[k])
    point_model, point_pers = compute_metrics(joined), compute_metrics(joined.assign(y_pred=joined["y_pers"]))
    lo, hi = np.percentile(diffs["macro_f1"], [2.5, 97.5])
    return dict(d_macro_f1=point_model["macro_f1"] - point_pers["macro_f1"],
                d_macro_f1_ci=f"[{lo:+.3f}, {hi:+.3f}]",
                p_better_macro_f1=float(np.mean(np.array(diffs["macro_f1"]) > 0)),
                d_warning_recall=point_model["warning_recall"] - point_pers["warning_recall"],
                p_better_warning_recall=float(np.mean(np.array(diffs["warning_recall"]) > 0)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base = load_predictions("persistence", "kb_only")
    rows = []
    for model, fs in CANDIDATES:
        if not (RESULTS_DIR / model / f"predictions__{fs}__forward.csv").exists():
            continue
        joined = load_predictions(model, fs).join(base[["y_pred"]].rename(columns={"y_pred": "y_pers"}))
        rows.append(dict(model=model, feature_set=fs, **bootstrap_difference(joined, rng)))
    res = pd.DataFrame(rows).sort_values("d_macro_f1", ascending=False)
    out = RESULTS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "bootstrap_vs_persistence.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
