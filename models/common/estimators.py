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
