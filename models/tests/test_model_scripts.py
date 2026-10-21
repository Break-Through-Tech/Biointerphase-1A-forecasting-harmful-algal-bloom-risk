"""Unit tests for the make_model function in each model folder."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODELS = Path(__file__).resolve().parents[1]


def load(folder):
    spec = importlib.util.spec_from_file_location(f"{folder}_train_test", MODELS / folder / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_data(n=160):
    idx = pd.date_range("2012-01-01", periods=n, freq="W-SUN")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"last_class": 0.0, "x": rng.normal(size=n), "woy_sin": 0.0, "woy_cos": 1.0}, index=idx)
    X.loc[X.index[::7], "x"] = np.nan                  # some missing values, as in the real table
    y = np.where(X["x"].fillna(0) > 1, 2, np.where(X["x"].fillna(0) > 0.5, 1, 0))
    return X, y


SINGLE_MODELS = ["persistence", "seasonal_climatology", "logistic_regression"]


@pytest.mark.parametrize("folder", SINGLE_MODELS)
def test_make_model_fits_and_predicts_three_classes(folder):
    X, y = small_data()
    model = load(folder).make_model().fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X) and set(pred) <= {0, 1, 2}


def test_logistic_regression_is_balanced_and_penalised():
    lr = load("logistic_regression").make_model()[-1]
    assert lr.class_weight == "balanced" and lr.C == 0.1
