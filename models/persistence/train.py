"""Baseline 1, persistence: next week's class = the last observed class.

Run from the models/ folder: python persistence/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.estimators import Persistence  # noqa: E402
from common.evaluation import run_model  # noqa: E402


def make_model() -> Persistence:
    """Return a new persistence forecast.

    Returns:
        Persistence: The model.
    """
    return Persistence()


if __name__ == "__main__":
    # Persistence reads only last_class, so one feature set is enough.
    run_model("persistence", make_model, feature_sets=["kb_only"])
