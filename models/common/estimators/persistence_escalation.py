"""Persistence plus escalation: persistence, with a learned model that can warn in quiet weeks.

Follows the scikit-learn interface (fit, predict, predict_proba, classes_), so
evaluation.cross_validate can run it like any other classifier.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score

from ..dates import assign_bloom_year

CLASSES = np.array([0, 1, 2])


class PersistencePlusEscalation(BaseEstimator, ClassifierMixin):
    """Persistence, plus a learned model that can raise a warning in quiet weeks.

    Persistence is hard to beat because the class rarely changes, but it can never
    forecast a change. This model keeps the last observed class, except when that
    class was no bloom and the base model's probability of at risk or bloom is at
    least a threshold; then it forecasts the more likely of at risk and bloom.

    The threshold is tuned only on the training years: each training bloom year is
    left out in turn, the base model is fitted on the rest, and the threshold that
    maximises F-beta for "quiet week turns" on those out-of-year predictions is kept.

    Args:
        base_factory (Callable): Returns a new, unfitted classifier with predict_proba.
        beta (float): F-beta weight used to pick the threshold (1 = F1).

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> idx = pd.date_range("2012-01-01", periods=240, freq="W-SUN")
        >>> rng = np.random.default_rng(0)
        >>> X = pd.DataFrame({"last_class": 0.0, "x": rng.normal(size=240)}, index=idx)
        >>> y = (X["x"] > 1).astype(int).to_numpy()            # quiet weeks turn when x is high
        >>> m = PersistencePlusEscalation(LogisticRegression).fit(X, y)
        >>> int(m.predict(X.assign(x=3.0)).min()), int(m.predict(X.assign(x=-3.0)).max())
        (1, 0)
    """

    def __init__(self, base_factory: Callable | None = None, beta: float = 1.0):
        self.base_factory = base_factory
        self.beta = beta

    @staticmethod
    def _warning_probability(model, X: pd.DataFrame) -> np.ndarray:
        """Return the probability of at risk or bloom.

        Args:
            model: Fitted classifier with predict_proba and classes_.
            X (pandas.DataFrame): Features.

        Returns:
            numpy.ndarray: P(at risk) + P(bloom) per row.
        """
        p, cls = model.predict_proba(X), list(model.classes_)
        return sum(p[:, cls.index(c)] for c in (1, 2) if c in cls)

    @staticmethod
    def _escalated_class(model, X: pd.DataFrame) -> np.ndarray:
        """Return bloom where P(bloom) >= P(at risk), else at risk.

        Args:
            model: Fitted classifier with predict_proba and classes_.
            X (pandas.DataFrame): Features.

        Returns:
            numpy.ndarray: 1 or 2 per row.
        """
        p, cls = model.predict_proba(X), list(model.classes_)
        p1 = p[:, cls.index(1)] if 1 in cls else 0 * p[:, 0]
        p2 = p[:, cls.index(2)] if 2 in cls else 0 * p[:, 0]
        return np.where(p2 >= p1, 2, 1)

    def fit(self, X: pd.DataFrame, y) -> "PersistencePlusEscalation":
        """Tune the escalation threshold on the training years, then fit the base model.

        Args:
            X (pandas.DataFrame): Features with a last_class column, indexed by week.
            y (array-like): Observed classes.

        Returns:
            PersistencePlusEscalation: self.
        """
        y = np.asarray(y)
        self.classes_ = CLASSES
        years = np.asarray(assign_bloom_year(X.index))
        quiet = (X["last_class"].fillna(0) == 0).to_numpy()
        oof = np.full(len(y), np.nan)
        for year in np.unique(years):
            te = years == year
            if (~te).sum() < 50 or len(np.unique(y[~te])) < 2:
                continue
            m = self.base_factory().fit(X[~te], y[~te])
            oof[te] = self._warning_probability(m, X[te])
        mask = quiet & ~np.isnan(oof)
        turned = y[mask] >= 1
        grid = np.linspace(0.05, 0.95, 37)
        scores = [fbeta_score(turned, oof[mask] >= t, beta=self.beta, zero_division=0) for t in grid]
        self.threshold_ = float(grid[int(np.argmax(scores))]) if turned.any() else 0.5
        self.base_ = self.base_factory().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the last observed class, escalated in quiet weeks above the threshold.

        Args:
            X (pandas.DataFrame): Features with a last_class column.

        Returns:
            numpy.ndarray: Forecast class per row.
        """
        last = X["last_class"].fillna(0).astype(int).to_numpy()
        escalate = (last == 0) & (self._warning_probability(self.base_, X) >= self.threshold_)
        return np.where(escalate, self._escalated_class(self.base_, X), last)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the base model's class probabilities, one column per class 0-2.

        Args:
            X (pandas.DataFrame): Features.

        Returns:
            numpy.ndarray: (n, 3) class probabilities.
        """
        p = self.base_.predict_proba(X)
        out = np.zeros((len(X), 3))
        for j, c in enumerate(self.base_.classes_):
            out[:, int(c)] = p[:, j]
        return out
