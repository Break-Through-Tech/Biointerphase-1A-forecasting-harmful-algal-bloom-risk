"""Unit tests for common/estimators.py."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.estimators import BalancedXGB, Persistence, PersistencePlusEscalation, SeasonalClimatology


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


def escalation_data(n=240):
    idx = pd.date_range("2012-01-01", periods=n, freq="W-SUN")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"last_class": 0.0, "x": rng.normal(size=n)}, index=idx)
    y = np.where(X["x"] > 1.5, 2, np.where(X["x"] > 1, 1, 0))
    return X, y


def test_escalation_only_changes_quiet_weeks():
    X, y = escalation_data()
    m = PersistencePlusEscalation(LogisticRegression).fit(X, y)
    assert 0.05 <= m.threshold_ <= 0.95
    assert (m.predict(X.assign(x=3.0)) >= 1).all()                    # quiet + high risk: warning
    assert (m.predict(X.assign(x=-3.0)) == 0).all()                   # quiet + low risk: stay quiet
    assert (m.predict(X.assign(x=3.0, last_class=1.0)) == 1).all()    # not quiet: persistence


def test_escalation_picks_the_more_likely_warning_class():
    X, y = escalation_data()
    m = PersistencePlusEscalation(LogisticRegression).fit(X, y)
    assert set(m.predict(X.assign(x=4.0))) == {2}
    assert m.predict_proba(X).shape == (len(X), 3)
