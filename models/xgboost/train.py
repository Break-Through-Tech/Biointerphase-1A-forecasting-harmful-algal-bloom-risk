"""XGBoost: gradient-boosted trees, each correcting the errors of the ones before.

Gradient boosting is a recommended method in the project brief, and Izadi et al. (2021)
found XGBoost outperformed random forest and SVM for forecasting Florida HAB onset.
Kept shallow and slow-learning because there are only about 740 labelled weeks.

Run from the models/ folder: python xgboost/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.estimators import BalancedXGB  # noqa: E402
from common.evaluation import run_model  # noqa: E402


def make_model() -> BalancedXGB:
    """Return a small, class-balanced XGBoost model.

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


if __name__ == "__main__":
    run_model("xgboost", make_model)
