"""Unit tests for common/estimators/persistence_escalation.py."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.estimators import PersistencePlusEscalation


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
