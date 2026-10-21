"""Unit tests for common/evaluation.py, on a small made-up weekly table."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from common import evaluation
from common.feature_sets import list_feature_columns


class RepeatLastClass:
    """Stand-in model: forecast the last observed class."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict(self, X):
        return X["last_class"].fillna(0).astype(int).to_numpy()

    def predict_proba(self, X):
        return np.eye(3)[self.predict(X)]


@pytest.fixture
def table():
    """Weekly table, Aug 2013 to Jul 2018, with the kb_only columns and a 1-week label."""
    idx = pd.date_range("2013-08-04", "2018-07-29", freq="W-SUN")
    rng = np.random.default_rng(0)
    t = pd.DataFrame(rng.normal(size=(len(idx), len(list_feature_columns("kb_only")))),
                     index=idx, columns=list_feature_columns("kb_only"))
    kb_class = pd.Series(np.repeat([0, 0, 2, 1, 0, 2], len(idx) // 6 + 1)[:len(idx)], index=idx, dtype=float)
    t["last_class"] = kb_class
    t["target"] = kb_class.shift(-1)
    t["target_week"] = idx + pd.Timedelta(weeks=1)
    return t


def test_cross_validate_returns_one_row_per_test_week(table):
    p = evaluation.cross_validate(lambda: DummyClassifier(strategy="most_frequent"), "kb_only", "forward", table)
    assert sorted(p["bloom_year"].unique()) == [2016, 2017, 2018]
    assert len(p) == table.loc[p.index.min():, "target"].notna().sum()
    assert list(p.columns) == ["bloom_year", "y_true", "y_pred", "p_no_bloom", "p_at_risk", "p_bloom", "last_class"]
    assert np.allclose(p[["p_no_bloom", "p_at_risk", "p_bloom"]].sum(axis=1), 1)


def test_cross_validate_with_a_repeat_last_class_model(table):
    p = evaluation.cross_validate(RepeatLastClass, "kb_only", "loyo", table)
    assert (p["y_pred"] == p["last_class"]).all()


def test_cross_validate_maps_probabilities_when_a_class_is_missing_in_training(table):
    t = table.assign(target=table["target"].replace(1.0, 0.0))     # no at-risk weeks at all
    p = evaluation.cross_validate(lambda: DummyClassifier(strategy="prior"), "kb_only", "forward", t)
    assert (p["p_at_risk"] == 0).all() and (p["p_bloom"] > 0).all()


def test_plots_write_png_files(table, tmp_path):
    p = evaluation.cross_validate(RepeatLastClass, "kb_only", "forward", table)
    evaluation.plot_timeline(p, "test", tmp_path / "timeline.png")
    evaluation.plot_confusion(p, "test", tmp_path / "confusion.png")
    assert (tmp_path / "timeline.png").stat().st_size > 0 and (tmp_path / "confusion.png").stat().st_size > 0


def test_run_model_saves_predictions_metrics_and_figures(table, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "build_weekly", lambda: (table, None))
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    summary = evaluation.run_model("persistence", RepeatLastClass, feature_sets=["kb_only"])
    out = tmp_path / "persistence"
    assert summary[["feature_set", "scheme"]].values.tolist() == [["kb_only", "loyo"], ["kb_only", "forward"]]
    for scheme in ("loyo", "forward"):
        for kind in ("predictions", "timeline", "confusion"):
            assert next(out.glob(f"{kind}__kb_only__{scheme}.*")).exists()
    assert pd.read_csv(out / "metrics.csv").shape[0] == 2
    assert len(json.loads((out / "metrics.json").read_text())) == 2
