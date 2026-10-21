"""Multinomial logistic regression: a linear model whose weights can be read as drivers.

The project brief asks for interpretable drivers (Biointerphase 1A, Challenge Project
Overview). Missing values are filled and features scaled inside each training fold,
so test years never influence either step.

Run from the models/ folder: python logistic_regression/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common.evaluation import run_model  # noqa: E402


def make_model(C: float = 0.1) -> Pipeline:
    """Return median imputation, scaling and a class-balanced logistic regression.

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


if __name__ == "__main__":
    run_model("logistic_regression", make_model)
