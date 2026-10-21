"""Unit tests for common/estimators.py."""
import numpy as np
import pandas as pd

from common.estimators import Persistence


def test_persistence_repeats_last_class_and_treats_missing_as_no_bloom():
    X = pd.DataFrame({"last_class": [2.0, 1.0, np.nan]})
    m = Persistence().fit(X, [0, 0, 0])
    assert m.predict(X).tolist() == [2, 1, 0]
    assert m.predict_proba(X).tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
