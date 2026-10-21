"""Train every model once on all labelled weeks and save it, so forecasts need no retraining.

Cross-validation trains a fresh model per test year and keeps only its predictions.
This script instead fits each model, on each of its feature sets, to every labelled
week and saves it with the feature columns it expects.

Run from the models/ folder, once per horizon:
    HAB_HORIZON=1 python save_final_models.py
Writes results/h<H>w/<model>/final__<feature_set>.joblib. Load one with:
    bundle = joblib.load("results/h1w/hybrid_lr/final__kb_only.joblib")
    bundle["model"].predict(weekly_table[bundle["features"]])
Loading needs the same scikit-learn and xgboost versions as requirements.txt.
"""
import importlib.util
from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd

from common.config import HORIZON_WEEKS, MODELS_ROOT, RESULTS_DIR
from common.feature_sets import FEATURE_SETS, list_feature_columns
from common.validation import select_labelled_rows
from common.weekly import build_weekly

ALL_SETS = list(FEATURE_SETS)
HYBRID_SETS = ["kb_only", "long", "full"]


def load_train_module(folder: str):
    """Load a model folder's train.py by path (a folder named xgboost would hide the package).

    Args:
        folder (str): Model folder under models/, e.g. "random_forest".

    Returns:
        module: The loaded train.py module.
    """
    spec = importlib.util.spec_from_file_location(f"{folder}_train_final", MODELS_ROOT / folder / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_models() -> dict[str, tuple[Callable, list[str]]]:
    """List every model to save with its factory and feature sets, matching the results folders.

    Returns:
        dict[str, tuple[Callable, list[str]]]: Results folder name -> (make_model, feature sets).
    """
    hybrid = load_train_module("hybrid_escalation")
    models = {"persistence": (load_train_module("persistence").make_model, ["kb_only"]),
              "seasonal_climatology": (load_train_module("seasonal_climatology").make_model, ["kb_only"])}
    for folder in ("logistic_regression", "random_forest", "xgboost"):
        models[folder] = (load_train_module(folder).make_model, ALL_SETS)
    models["hybrid_lr"] = (lambda: hybrid.make_hybrid("logistic_regression"), HYBRID_SETS)
    models["hybrid_xgb"] = (lambda: hybrid.make_hybrid("xgboost"), HYBRID_SETS)
    return models


def fit_final_model(make_model: Callable, table: pd.DataFrame, feature_set: str) -> dict:
    """Fit a model on every labelled week and bundle it with what is needed to use it.

    Args:
        make_model (Callable): Returns a new, unfitted classifier.
        table (pandas.DataFrame): Weekly table with a "target" column.
        feature_set (str): A key of FEATURE_SETS.

    Returns:
        dict: model (fitted), features (column order it expects), feature_set,
        horizon_weeks, trained_on_weeks and last_training_week.
    """
    lab = select_labelled_rows(table)
    cols = list_feature_columns(feature_set)
    model = make_model().fit(lab[cols], lab["target"].to_numpy())
    if hasattr(model, "base_factory"):
        # The hybrid keeps the function that builds its base model, only needed to refit.
        # It cannot be saved, so it is dropped; the fitted base model is kept in base_.
        model.base_factory = None
    return dict(model=model, features=cols, feature_set=feature_set, horizon_weeks=HORIZON_WEEKS,
                trained_on_weeks=int(len(lab)), last_training_week=str(lab.index.max().date()))


if __name__ == "__main__":
    table, _ = build_weekly()
    for name, (make_model, sets) in list_models().items():
        for fs in sets:
            bundle = fit_final_model(make_model, table, fs)
            path = Path(RESULTS_DIR / name / f"final__{fs}.joblib")
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path, compress=3)
            print(f"{path.relative_to(MODELS_ROOT)}: {path.stat().st_size / 1024:.0f} KB")
