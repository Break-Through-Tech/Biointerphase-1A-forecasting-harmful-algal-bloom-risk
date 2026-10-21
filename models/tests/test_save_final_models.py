"""Unit tests for save_final_models.py."""
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.estimators import PersistencePlusEscalation
from common.feature_sets import list_feature_columns

MODELS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("save_final_models_test", MODELS / "save_final_models.py")
sfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfm)


def table(n=260):
    idx = pd.date_range("2012-01-01", periods=n, freq="W-SUN")
    rng = np.random.default_rng(0)
    t = pd.DataFrame(rng.normal(size=(n, len(list_feature_columns("kb_only")))),
                     index=idx, columns=list_feature_columns("kb_only"))
    t["last_class"] = 0.0
    t["target"] = np.where(t["kb_max_log"] > 1, 2, np.where(t["kb_max_log"] > 0.5, 1, 0)).astype(float)
    t.iloc[-1, t.columns.get_loc("target")] = np.nan          # last week has no label yet
    return t


def test_fit_final_model_uses_every_labelled_week_and_records_the_features():
    bundle = sfm.fit_final_model(LogisticRegression, table(), "kb_only")
    assert bundle["trained_on_weeks"] == 259
    assert bundle["features"] == list_feature_columns("kb_only")
    assert bundle["last_training_week"] == "2016-12-11"
    assert set(bundle["model"].predict(table()[bundle["features"]])) <= {0, 1, 2}


def test_fit_final_model_makes_the_hybrid_saveable(tmp_path):
    bundle = sfm.fit_final_model(lambda: PersistencePlusEscalation(LogisticRegression), table(), "kb_only")
    assert bundle["model"].base_factory is None
    joblib.dump(bundle, tmp_path / "hybrid.joblib")
    loaded = joblib.load(tmp_path / "hybrid.joblib")
    X = table()[bundle["features"]]
    assert (loaded["model"].predict(X) == bundle["model"].predict(X)).all()


def test_list_models_matches_the_results_folders():
    models = sfm.list_models()
    assert list(models) == ["persistence", "seasonal_climatology", "logistic_regression", "random_forest",
                            "xgboost", "hybrid_lr", "hybrid_xgb"]
    assert models["persistence"][1] == ["kb_only"] and models["hybrid_lr"][1] == ["kb_only", "long", "full"]
