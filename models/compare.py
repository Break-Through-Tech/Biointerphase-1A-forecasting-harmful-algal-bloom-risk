"""Collect every model's scores into one comparison table.

Run from the models/ folder after the models: python compare.py
Writes results/comparison.csv and prints the forward-chaining scores per horizon.
"""
from pathlib import Path

import pandas as pd

from common.config import MODELS_ROOT

COLUMNS = ["horizon", "model", "feature_set", "scheme", "macro_f1", "kappa", "balanced_accuracy",
           "recall_at_risk", "recall_bloom", "precision_bloom", "warning_recall", "warning_precision",
           "n_onsets", "onsets_caught_bloom", "onsets_caught_any_warning",
           "n_escalations", "escalation_recall", "escalation_precision", "roc_auc_ovr", "pr_auc_bloom"]


def collect_metrics(results_root: Path) -> pd.DataFrame:
    """Stack every results/h*w/<model>/metrics.csv into one table.

    Args:
        results_root (pathlib.Path): The results/ folder.

    Returns:
        pandas.DataFrame: One row per horizon, model, feature set and scheme, with COLUMNS.
    """
    frames = [pd.read_csv(f).assign(horizon=f.parent.parent.name)
              for f in sorted(results_root.glob("h*w/*/metrics.csv"))]
    return pd.concat(frames, ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    results = MODELS_ROOT / "results"
    table = collect_metrics(results)
    table.to_csv(results / "comparison.csv", index=False)
    pd.set_option("display.width", 250)
    for h, g in table[table["scheme"] == "forward"].groupby("horizon"):
        print(f"\n##### {h} (forward-chaining) #####")
        print(g.drop(columns=["horizon", "scheme"]).sort_values("macro_f1", ascending=False)
              .round(3).to_string(index=False))
