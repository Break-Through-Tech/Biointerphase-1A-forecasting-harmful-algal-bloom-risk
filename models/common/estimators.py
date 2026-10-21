"""Forecast models that are not plain scikit-learn objects.

Each follows the scikit-learn interface (fit, predict, predict_proba, classes_), so
evaluation.cross_validate can run them like any other classifier.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score
from sklearn.utils.class_weight import compute_sample_weight

from .dates import assign_bloom_year

CLASSES = np.array([0, 1, 2])


class Persistence(BaseEstimator, ClassifierMixin):
    """Forecast that the next class equals the last observed class.

    Red tide changes slowly from week to week, so this is the baseline every learned
    model has to beat (Medina et al. 2024 compare their model against the same
    idea). It needs only the last_class feature and learns nothing.

    Example:
        >>> X = pd.DataFrame({"last_class": [0, 2, None]})
        >>> Persistence().fit(X, [0, 2, 0]).predict(X).tolist()
        [0, 2, 0]
    """

    def fit(self, X: pd.DataFrame, y) -> "Persistence":
        """Set the classes; there is nothing to learn.

        Args:
            X (pandas.DataFrame): Features (unused).
            y (array-like): Labels (unused).

        Returns:
            Persistence: self.
        """
        self.classes_ = CLASSES
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the last observed class (no bloom if nothing has been observed yet).

        Args:
            X (pandas.DataFrame): Features with a last_class column.

        Returns:
            numpy.ndarray: Forecast class per row.
        """
        return X["last_class"].fillna(0).astype(int).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability 1 for the forecast class and 0 for the others.

        Args:
            X (pandas.DataFrame): Features with a last_class column.

        Returns:
            numpy.ndarray: (n, 3) class probabilities.
        """
        return np.eye(3)[self.predict(X)]


class SeasonalClimatology(BaseEstimator, ClassifierMixin):
    """Forecast from how often each class occurred at this time of year.

    For each week of the year, counts the classes in the training weeks within
    `window` weeks of it, and divides by each class's overall share (the same
    correction as class_weight="balanced"; without it the model would always say no
    bloom). It uses no information about the current year.

    Args:
        window (int): Weeks either side of the week of year to pool.

    Example:
        >>> week = np.tile(np.arange(52), 4)                    # four years of weeks
        >>> X = pd.DataFrame({"woy_sin": np.sin(week / 52 * 2 * np.pi), "woy_cos": np.cos(week / 52 * 2 * np.pi)})
        >>> y = np.select([week >= 39, week >= 35], [2, 1], 0)  # at risk, then blooms, late each year
        >>> m = SeasonalClimatology(window=2).fit(X, y)
        >>> m.predict(X.iloc[[10, 45]]).tolist()
        [0, 2]
    """

    def __init__(self, window: int = 3):
        self.window = window

    @staticmethod
    def _week_of_year(X: pd.DataFrame) -> np.ndarray:
        """Recover the week of year (0-51) from its sine and cosine.

        Args:
            X (pandas.DataFrame): Features with woy_sin and woy_cos columns.

        Returns:
            numpy.ndarray: Week of year per row.
        """
        angle = np.arctan2(X["woy_sin"].to_numpy(), X["woy_cos"].to_numpy())
        return (np.round(angle / (2 * np.pi) * 52.18) % 52).astype(int)

    def fit(self, X: pd.DataFrame, y) -> "SeasonalClimatology":
        """Build the table of class probabilities for each week of the year.

        Args:
            X (pandas.DataFrame): Features with woy_sin and woy_cos columns.
            y (array-like): Observed classes.

        Returns:
            SeasonalClimatology: self.
        """
        self.classes_ = CLASSES
        woy, y = self._week_of_year(X), np.asarray(y)
        prior = np.array([(y == c).mean() + 1e-6 for c in self.classes_])
        self.table_ = np.zeros((52, 3))
        for w in range(52):
            dist = np.minimum(np.abs(woy - w), 52 - np.abs(woy - w))
            near = y[dist <= self.window]
            counts = np.array([(near == c).sum() for c in self.classes_]) + 1.0   # add-one smoothing
            p = (counts / counts.sum()) / prior
            self.table_[w] = p / p.sum()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Look up the class probabilities for each row's week of year.

        Args:
            X (pandas.DataFrame): Features with woy_sin and woy_cos columns.

        Returns:
            numpy.ndarray: (n, 3) class probabilities.
        """
        return self.table_[self._week_of_year(X)]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the most likely class for each row's week of year.

        Args:
            X (pandas.DataFrame): Features with woy_sin and woy_cos columns.

        Returns:
            numpy.ndarray: Forecast class per row.
        """
        return self.predict_proba(X).argmax(axis=1)


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost with balanced sample weights that tolerates folds missing a class.

    XGBoost needs labels 0..k-1 and has no class_weight option, so labels are
    re-indexed to the classes present in training and each week is weighted by the
    inverse frequency of its class.

    Args:
        **params: Passed to xgboost.XGBClassifier.

    Example:
        >>> X = pd.DataFrame({"x": np.r_[np.zeros(20), np.ones(20)]})
        >>> y = np.r_[np.zeros(20, int), np.full(20, 2)]          # a fold with no at-risk weeks
        >>> m = BalancedXGB(n_estimators=20, max_depth=2).fit(X, y)
        >>> m.classes_.tolist(), m.predict(X.iloc[[0, 39]]).tolist()
        ([0, 2], [0, 2])
    """

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep: bool = True) -> dict:
        """Return the XGBoost parameters.

        Args:
            deep (bool): Unused; kept for the scikit-learn interface.

        Returns:
            dict: The parameters passed to XGBClassifier.
        """
        return dict(self.params)

    def set_params(self, **params) -> "BalancedXGB":
        """Update the XGBoost parameters.

        Args:
            **params: Parameters to change.

        Returns:
            BalancedXGB: self.
        """
        self.params.update(params)
        return self

    def fit(self, X: pd.DataFrame, y) -> "BalancedXGB":
        """Fit XGBoost with balanced sample weights.

        Args:
            X (pandas.DataFrame): Features (NaN allowed).
            y (array-like): Observed classes.

        Returns:
            BalancedXGB: self.
        """
        from xgboost import XGBClassifier
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        self.model_ = XGBClassifier(**self.params)
        self.model_.fit(X, np.array([index[v] for v in y]), sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, one column per class in classes_.

        Args:
            X (pandas.DataFrame): Features.

        Returns:
            numpy.ndarray: (n, len(classes_)) class probabilities.
        """
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the most likely class.

        Args:
            X (pandas.DataFrame): Features.

        Returns:
            numpy.ndarray: Forecast class per row.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


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
