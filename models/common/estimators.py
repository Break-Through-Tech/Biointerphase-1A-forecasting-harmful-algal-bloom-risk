"""Forecast models that are not plain scikit-learn objects.

Each follows the scikit-learn interface (fit, predict, predict_proba, classes_), so
evaluation.cross_validate can run them like any other classifier.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


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
