"""Random forest: many decision trees voting on the class.

A recommended method in the project brief, and the model in Medina et al.'s (2024)
K. brevis early-warning forecast. Trees capture thresholds and interactions, need no
scaling, and scikit-learn trees accept missing values directly.

Run from the models/ folder: python random_forest/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from common.evaluation import run_model  # noqa: E402


def make_model() -> RandomForestClassifier:
    """Return a class-balanced random forest sized for about 740 labelled weeks.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The unfitted model.
    """
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,          # small data: no single-week leaves
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


if __name__ == "__main__":
    run_model("random_forest", make_model)
