"""Persistence plus escalation: keep last week's class unless a learned model flags a change.

Two versions: a logistic regression base (Hybrid LR) and an XGBoost base (Hybrid XGB).
The escalation threshold is tuned on the training years only (see
common/estimators/persistence_escalation.py). Settings are read from
models/model_config.toml.

Run from the models/ folder, for one horizon (HAB_HORIZON, default 1):
    python hybrid_escalation/train.py                 # both hybrids
    python hybrid_escalation/train.py hybrid_xgb      # one hybrid
To run every horizon and retrain the saved models, use run_pipeline.py.
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
from common.model_config import load_model_config  # noqa: E402


def make_logistic_regression() -> Pipeline:
    """Return median imputation, scaling and a class-balanced logistic regression.

    Missing values are filled and features scaled inside each training fold, so test
    years never influence either step. C and max_iter come from the
    [logistic_regression] section of model_config.toml.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model.
    """
    return make_pipeline(
        # add_indicator: "this source was missing this week" can itself be informative.
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", **load_model_config()["logistic_regression"]),
    )


def make_xgboost() -> BalancedXGB:
    """Return a small, class-balanced XGBoost model.

    Settings come from the [xgboost] section of model_config.toml.

    Returns:
        BalancedXGB: The unfitted model.
    """
    return BalancedXGB(objective="multi:softprob", eval_metric="mlogloss", **load_model_config()["xgboost"])


def make_hybrid_lr() -> PersistencePlusEscalation:
    """Return persistence plus escalation with a logistic regression base (Hybrid LR).

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(make_logistic_regression, beta=load_model_config()["hybrid"]["beta"])


def make_hybrid_xgb() -> PersistencePlusEscalation:
    """Return persistence plus escalation with an XGBoost base (Hybrid XGB).

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(make_xgboost, beta=load_model_config()["hybrid"]["beta"])


HYBRIDS = {"hybrid_lr": make_hybrid_lr, "hybrid_xgb": make_hybrid_xgb}


if __name__ == "__main__":
    for name in sys.argv[1:] or HYBRIDS:
        run_model(name, HYBRIDS[name], feature_sets=load_model_config()["run"]["feature_sets"])
