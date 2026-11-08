"""Persistence plus escalation: keep last week's class unless a learned model flags a change.

Two versions: a logistic regression base (Hybrid LR) and an XGBoost base (Hybrid XGB).
The escalation threshold is tuned on the training years only (see
common/estimators/persistence_escalation.py).

Run from the models/ folder: python hybrid_escalation/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common.estimators import BalancedXGB, PersistencePlusEscalation  # noqa: E402
from common.evaluation import run_model  # noqa: E402

FEATURE_SETS = ["kb_only", "long", "full"]


def make_logistic_regression(C: float = 0.1) -> Pipeline:
    """Return median imputation, scaling and a class-balanced logistic regression.

    Missing values are filled and features scaled inside each training fold, so test
    years never influence either step.

    Args:
        C (float): Inverse L2 penalty strength. 0.1 is a strong penalty, because there
            are about 740 labelled weeks and up to 60 correlated features.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model.
    """
    return make_pipeline(
        # add_indicator: "this source was missing this week" can itself be informative.
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
    )


def make_xgboost() -> BalancedXGB:
    """Return a small, class-balanced XGBoost model.

    Kept shallow and slow-learning (300 trees of depth 3, learning rate 0.05) because
    there are only about 740 labelled weeks.

    Returns:
        BalancedXGB: The unfitted model.
    """
    return BalancedXGB(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=4,
        random_state=42,
    )


def make_hybrid_lr() -> PersistencePlusEscalation:
    """Return persistence plus escalation with a logistic regression base (Hybrid LR).

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(make_logistic_regression)


def make_hybrid_xgb() -> PersistencePlusEscalation:
    """Return persistence plus escalation with an XGBoost base (Hybrid XGB).

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(make_xgboost)


if __name__ == "__main__":
    run_model("hybrid_lr", make_hybrid_lr, feature_sets=FEATURE_SETS)
    run_model("hybrid_xgb", make_hybrid_xgb, feature_sets=FEATURE_SETS)
