"""Unit tests for common/estimators.py."""
import numpy as np
import pandas as pd

from common.estimators import Persistence, SeasonalClimatology


def test_persistence_repeats_last_class_and_treats_missing_as_no_bloom():
    X = pd.DataFrame({"last_class": [2.0, 1.0, np.nan]})
    m = Persistence().fit(X, [0, 0, 0])
    assert m.predict(X).tolist() == [2, 1, 0]
    assert m.predict_proba(X).tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def season_frame():
    woy = np.tile(np.arange(52), 4)
    angle = woy / 52 * 2 * np.pi
    X = pd.DataFrame({"woy_sin": np.sin(angle), "woy_cos": np.cos(angle)})
    y = np.select([woy >= 39, woy >= 35], [2, 1], 0)   # at risk, then blooms, late in every year
    return X, y


def test_seasonal_climatology_learns_the_bloom_season():
    X, y = season_frame()
    m = SeasonalClimatology(window=2).fit(X, y)
    assert m.predict(X.iloc[[5, 45]]).tolist() == [0, 2]
    assert np.allclose(m.predict_proba(X).sum(axis=1), 1)


def test_seasonal_climatology_recovers_week_of_year():
    X, _ = season_frame()
    assert SeasonalClimatology._week_of_year(X).tolist()[:52] == list(range(52))
