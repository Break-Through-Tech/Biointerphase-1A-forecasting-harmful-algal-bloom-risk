"""Baseline 2, seasonal climatology: how bloom-prone is this week of the year?

Run from the models/ folder: python seasonal_climatology/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.estimators import SeasonalClimatology  # noqa: E402
from common.evaluation import run_model  # noqa: E402


def make_model() -> SeasonalClimatology:
    """Return a new seasonal climatology that pools 3 weeks either side.

    Returns:
        SeasonalClimatology: The model.
    """
    return SeasonalClimatology(window=3)


if __name__ == "__main__":
    # The climatology reads only the week of year, so one feature set is enough.
    run_model("seasonal_climatology", make_model, feature_sets=["kb_only"])
