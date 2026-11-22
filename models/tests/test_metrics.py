"""Unit tests for common/metrics.py."""
import numpy as np
import pandas as pd
import pytest

from common.metrics import compute_scores, find_bloom_onsets, score_onsets, summarise_predictions

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


def test_score_onsets_counts_bloom_and_any_warning():
    p = pd.DataFrame({"y_true": [0, 2, 0, 0, 0, 0, 0, 2], "y_pred": [0, 2, 0, 0, 0, 0, 0, 1]}, index=WEEKS)
    assert score_onsets(p) == {"n_onsets": 2, "onsets_caught_bloom": 1, "onsets_caught_any_warning": 2}


def test_summarise_predictions_scores_escalations_in_quiet_weeks_only():
    p = pd.DataFrame({"y_true": [0, 1, 2, 2], "y_pred": [0, 0, 2, 2], "last_class": [0, 0, 0, 2],
                      "p_no_bloom": [.8, .6, .1, .1], "p_at_risk": [.1, .3, .2, .2], "p_bloom": [.1, .1, .7, .7]},
                     index=WEEKS[:4])
    s = summarise_predictions(p)
    assert s["n_quiet_weeks"] == 3 and s["n_escalations"] == 2
    assert s["escalation_recall"] == 0.5 and s["escalation_precision"] == 1.0
    assert s["agrees_with_persistence"] == 0.75
