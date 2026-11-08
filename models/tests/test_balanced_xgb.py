"""Unit tests for common/estimators/balanced_xgb.py."""
import numpy as np
import pandas as pd

from common.estimators import BalancedXGB


def test_balanced_xgb_handles_folds_missing_a_class():
    X = pd.DataFrame({"x": np.r_[np.zeros(20), np.ones(20)]})
    y = np.r_[np.zeros(20, int), np.full(20, 2)]                  # no at-risk weeks in this fold
    m = BalancedXGB(n_estimators=20, max_depth=2).fit(X, y)
    assert m.classes_.tolist() == [0, 2]
    assert m.predict(X).tolist() == y.tolist()
    assert m.predict_proba(X).shape == (40, 2)


def test_balanced_xgb_get_and_set_params():
    m = BalancedXGB(max_depth=2)
    assert m.set_params(max_depth=4).get_params() == {"max_depth": 4}
