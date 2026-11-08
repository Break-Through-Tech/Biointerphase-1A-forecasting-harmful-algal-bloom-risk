"""Cross-validate a model on the weekly table."""
from collections.abc import Callable

import numpy as np
import pandas as pd

from .feature_sets import list_feature_columns
from .validation import make_folds, select_labelled_rows
from .weekly import build_weekly


def cross_validate(make_model: Callable, feature_set: str, scheme: str,
                   table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit a fresh model on each fold and collect its out-of-fold predictions.

    Args:
        make_model (Callable): Returns a new, unfitted scikit-learn style classifier.
        feature_set (str): A key of feature_sets.FEATURE_SETS, e.g. "kb_only".
        scheme (str): "forward", "loyo" or "holdout" (see validation.make_folds).
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
    for _, tr, te in make_folds(lab, scheme):
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
            "bloom_year": lab["bloom_year"].to_numpy()[te], "y_true": lab["target"].to_numpy()[te], "y_pred": np.asarray(pred).astype(int),
            "p_no_bloom": proba[:, 0], "p_at_risk": proba[:, 1], "p_bloom": proba[:, 2],
            "last_class": lab["last_class"].to_numpy()[te],
        }, index=lab.index[te]))
    return pd.concat(rows).sort_index()
