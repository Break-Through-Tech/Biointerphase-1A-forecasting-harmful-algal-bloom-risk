"""Unit tests for common/feature_sets.py."""
import pytest

from common.feature_sets import FEATURE_GROUPS, FEATURE_SETS, NUTRIENT_EXTRA, list_feature_columns


def test_list_feature_columns_sizes():
    assert len(list_feature_columns("kb_only")) == 23
    assert len(list_feature_columns("full")) == 62


def test_list_feature_columns_follows_group_order():
    cols = list_feature_columns("kb_only")
    assert cols[:2] == FEATURE_GROUPS["season"]
    assert cols[2:] == FEATURE_GROUPS["kb_lags"]


def test_feature_sets_use_known_groups_and_no_duplicate_columns():
    for name, groups in FEATURE_SETS.items():
        assert set(groups) <= set(FEATURE_GROUPS)
        cols = list_feature_columns(name)
        assert len(cols) == len(set(cols))


def test_extra_water_quality_columns_are_in_no_feature_set():
    used = {g for groups in FEATURE_SETS.values() for g in groups}
    assert "water_quality_extra" not in used
    assert FEATURE_GROUPS["water_quality_extra"] == [f"{c}_8w" for c in NUTRIENT_EXTRA]


def test_list_feature_columns_rejects_unknown_sets():
    with pytest.raises(KeyError):
        list_feature_columns("not_a_set")
