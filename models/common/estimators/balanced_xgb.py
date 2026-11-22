"""XGBoost with balanced sample weights, for folds that may miss a class.

Follows the scikit-learn interface (fit, predict, predict_proba, classes_), so
evaluation.cross_validate can run it like any other classifier.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

CLASSES = np.array([0, 1, 2])


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
