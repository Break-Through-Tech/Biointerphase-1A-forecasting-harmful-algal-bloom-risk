"""Unit tests for common/metrics.py."""
import numpy as np
import pandas as pd
import pytest

from common.metrics import compute_scores, find_bloom_onsets

WEEKS = pd.date_range("2020-01-05", periods=8, freq="W-SUN")


def test_find_bloom_onsets_needs_four_quiet_weeks_between_blooms():
    y = pd.Series([0, 2, 2, 0, 0, 0, 0, 2], index=WEEKS)
    assert find_bloom_onsets(y) == [WEEKS[1], WEEKS[7]]
    assert find_bloom_onsets(y, quiet_weeks=5) == [WEEKS[1]]


def test_find_bloom_onsets_ignores_at_risk_weeks():
    assert find_bloom_onsets(pd.Series([0, 1, 1, 0], index=WEEKS[:4])) == []


def test_compute_scores_perfect_forecast():
    y = pd.Series([0, 1, 2, 0])
    s = compute_scores(y, y)
    assert s["accuracy"] == s["macro_f1"] == s["kappa"] == 1.0
    assert s["n"] == 4 and "roc_auc_ovr" not in s


def test_compute_scores_counts_any_warning_as_caught():
    s = compute_scores(pd.Series([0, 1, 2, 2]), pd.Series([0, 2, 1, 0]))
    assert s["recall_bloom"] == 0.0
    assert s["warning_recall"] == pytest.approx(2 / 3)
    assert s["warning_precision"] == 1.0


def test_compute_scores_adds_ranking_scores_when_probabilities_are_given():
    y = pd.Series([0, 1, 2])
    s = compute_scores(y, y, np.eye(3))
    assert s["roc_auc_ovr"] == 1.0 and s["pr_auc_bloom"] == 1.0
    assert "roc_auc_ovr" not in compute_scores(y, y, np.full((3, 3), np.nan))
