"""Persistence plus escalation: keep last week's class unless a learned model flags a change.

Two versions: a logistic regression base (Hybrid LR) and an XGBoost base (Hybrid XGB).
The escalation threshold is tuned on the training years only (see
common/estimators.PersistencePlusEscalation).

Run from the models/ folder: python hybrid_escalation/train.py
"""
import importlib.util
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.estimators import PersistencePlusEscalation  # noqa: E402
from common.evaluation import run_model  # noqa: E402

FEATURE_SETS = ["kb_only", "long", "full"]


@lru_cache
def load_make_model(folder: str) -> Callable:
    """Load make_model from another model's train.py.

    The folders are loaded by path rather than imported, because a folder named
    xgboost would hide the xgboost package. Each folder is loaded once.

    Args:
        folder (str): Model folder under models/, e.g. "logistic_regression".

    Returns:
        Callable: That folder's make_model function.
    """
    path = Path(__file__).resolve().parents[1] / folder / "train.py"
    spec = importlib.util.spec_from_file_location(f"{folder}_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_model


def make_hybrid(base: str) -> PersistencePlusEscalation:
    """Return a persistence-plus-escalation model on top of another model.

    Args:
        base (str): Model folder of the base model, "logistic_regression" or "xgboost".

    Returns:
        PersistencePlusEscalation: The unfitted model.
    """
    return PersistencePlusEscalation(load_make_model(base))


if __name__ == "__main__":
    run_model("hybrid_lr", lambda: make_hybrid("logistic_regression"), feature_sets=FEATURE_SETS)
    run_model("hybrid_xgb", lambda: make_hybrid("xgboost"), feature_sets=FEATURE_SETS)
