"""Unit tests for hybrid_escalation/train.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASES = ["make_logistic_regression"]
HYBRIDS = ["make_hybrid_lr"]
MODELS = Path(__file__).resolve().parents[1]


def load_train_script():
    spec = importlib.util.spec_from_file_location("hybrid_train_test", MODELS / "hybrid_escalation" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_data(n=160):
    idx = pd.date_range("2012-01-01", periods=n, freq="W-SUN")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"last_class": 0.0, "x": rng.normal(size=n)}, index=idx)
    X.loc[X.index[::7], "x"] = np.nan                  # some missing values, as in the real table
    y = np.where(X["x"].fillna(0) > 1, 2, np.where(X["x"].fillna(0) > 0.5, 1, 0))
    return X, y


@pytest.mark.parametrize("base", BASES)
def test_base_model_fits_with_missing_values(base):
    X, y = small_data()
    p = getattr(load_train_script(), base)().fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize("hybrid", HYBRIDS)
def test_hybrid_forecasts_three_classes(hybrid):
    X, y = small_data()
    m = getattr(load_train_script(), hybrid)().fit(X, y)
    assert set(m.predict(X)) <= {0, 1, 2}
    assert (m.predict(X.assign(last_class=2.0)) == 2).all()     # not quiet: persistence
