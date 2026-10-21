"""Cross-validate a model on the weekly table, and save its predictions, scores and plots."""
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # save figures without opening a window
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .config import CLASS_NAMES, RESULTS_DIR  # noqa: E402
from .feature_sets import FEATURE_SETS, list_feature_columns  # noqa: E402
from .metrics import find_bloom_onsets, summarise_predictions  # noqa: E402
from .validation import make_folds, select_labelled_rows  # noqa: E402
from .weekly import build_weekly  # noqa: E402

SUMMARY_COLUMNS = ["feature_set", "scheme", "macro_f1", "kappa", "recall_at_risk", "recall_bloom",
                   "warning_recall", "onsets_caught_bloom", "onsets_caught_any_warning",
                   "n_escalations", "escalation_recall", "escalation_precision"]


def cross_validate(make_model: Callable, feature_set: str, scheme: str,
                   table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit a fresh model on each fold and collect its out-of-fold predictions.

    Args:
        make_model (Callable): Returns a new, unfitted scikit-learn style classifier.
        feature_set (str): A key of feature_sets.FEATURE_SETS, e.g. "kb_only".
        scheme (str): "loyo" or "forward" (see validation.make_folds).
        table (pandas.DataFrame | None): Weekly table; built with weekly.build_weekly if None.

    Returns:
        pandas.DataFrame: One row per test week: bloom_year, y_true, y_pred, the three
        class probabilities (NaN if the model has no predict_proba) and last_class.

    Example:
        >>> from sklearn.dummy import DummyClassifier
        >>> p = cross_validate(lambda: DummyClassifier(strategy="most_frequent"), "kb_only", "forward")
        >>> p["y_pred"].unique().tolist(), int(p["bloom_year"].min())
        ([0], 2016)
    """
    table = build_weekly()[0] if table is None else table
    lab = select_labelled_rows(table)
    cols = list_feature_columns(feature_set)
    rows = []
    for year, tr, te in make_folds(lab, scheme):
        model = make_model()
        model.fit(lab[cols].iloc[tr], lab["target"].to_numpy()[tr])
        X_te = lab[cols].iloc[te]
        pred = model.predict(X_te)
        proba = np.full((te.sum(), 3), np.nan)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_te)
            # A fold missing a class in training gives fewer columns; map them back by class.
            classes = list(getattr(model, "classes_", range(p.shape[1])))
            proba = np.zeros((te.sum(), 3))
            for j, c in enumerate(classes):
                proba[:, int(c)] = p[:, j]
        rows.append(pd.DataFrame({
            "bloom_year": year, "y_true": lab["target"].to_numpy()[te], "y_pred": np.asarray(pred).astype(int),
            "p_no_bloom": proba[:, 0], "p_at_risk": proba[:, 1], "p_bloom": proba[:, 2],
            "last_class": lab["last_class"].to_numpy()[te],
        }, index=lab.index[te]))
    return pd.concat(rows).sort_index()


def plot_timeline(predictions: pd.DataFrame, title: str, path: Path) -> None:
    """Save a plot of observed and forecast classes over time, with bloom onsets marked.

    Args:
        predictions (pandas.DataFrame): Output of cross_validate.
        title (str): Figure title.
        path (pathlib.Path): PNG file to write.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.scatter(predictions.index, predictions["y_true"] + 0.08, s=10, c="0.4", label="observed")
    ax.scatter(predictions.index, predictions["y_pred"] - 0.08, s=10, c="tab:red", label="forecast")
    for week in find_bloom_onsets(predictions["y_true"]):
        ax.axvline(week, color="tab:orange", lw=0.8, alpha=0.6)
    ax.set_yticks([0, 1, 2], CLASS_NAMES)
    ax.set_title(title + "  (orange lines = bloom onsets)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(predictions: pd.DataFrame, title: str, path: Path) -> None:
    """Save a confusion matrix of observed against forecast classes.

    Args:
        predictions (pandas.DataFrame): Output of cross_validate.
        title (str): Figure title.
        path (pathlib.Path): PNG file to write.

    Returns:
        None
    """
    cm = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1, 2], CLASS_NAMES, rotation=20)
    ax.set_yticks([0, 1, 2], CLASS_NAMES)
    ax.set_xlabel("forecast")
    ax.set_ylabel("observed")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_model(model_name: str, make_model: Callable, feature_sets: Iterable[str] | None = None,
              schemes: Iterable[str] = ("loyo", "forward")) -> pd.DataFrame:
    """Cross-validate one model on each feature set and scheme, and save everything.

    Writes to RESULTS_DIR/<model_name>/: predictions__<set>__<scheme>.csv,
    timeline__<set>__<scheme>.png, confusion__<set>__<scheme>.png, metrics.csv and
    metrics.json.

    Args:
        model_name (str): Folder name for the results, e.g. "persistence".
        make_model (Callable): Returns a new, unfitted classifier.
        feature_sets (Iterable[str] | None): Feature sets to run; all of FEATURE_SETS if None.
        schemes (Iterable[str]): Validation schemes to run.

    Returns:
        pandas.DataFrame: One row of scores per feature set and scheme.
    """
    table, _ = build_weekly()
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for fs in feature_sets or FEATURE_SETS:
        for scheme in schemes:
            pred = cross_validate(make_model, fs, scheme, table)
            tag = f"{fs}__{scheme}"
            pred.to_csv(out_dir / f"predictions__{tag}.csv")
            summary.append(dict(model=model_name, feature_set=fs, scheme=scheme, **summarise_predictions(pred)))
            plot_timeline(pred, f"{model_name} | {fs} | {scheme}", out_dir / f"timeline__{tag}.png")
            plot_confusion(pred, f"{model_name}\n{fs} | {scheme}", out_dir / f"confusion__{tag}.png")
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(summary.to_dict(orient="records"), fh, indent=2, default=float)
    print(f"\n=== {model_name} ===")
    print(summary[SUMMARY_COLUMNS].round(3).to_string(index=False))
    return summary
