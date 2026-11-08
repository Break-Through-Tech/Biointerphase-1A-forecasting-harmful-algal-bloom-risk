"""Persistence plus escalation: keep last week's class unless a learned model flags a change.

Hybrid LR: the learned model is a class-balanced logistic regression. The escalation
threshold is tuned on the training years only (see
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

from common.estimators import PersistencePlusEscalation  # noqa: E402
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


def make_hybrid_lr() -> PersistencePlusEscalation:
    """Return persistence plus escalation with a logistic regression base (Hybrid LR).

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(make_logistic_regression)


if __name__ == "__main__":
    run_model("hybrid_lr", make_hybrid_lr, feature_sets=FEATURE_SETS)
