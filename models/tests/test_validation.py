"""Unit tests for common/validation.py."""
import pandas as pd
import pytest

from common.validation import FORWARD_FIRST_TEST_YEAR, make_folds, select_labelled_rows


def weekly_table(start="2013-08-04", end="2018-07-29", horizon=1):
    """A made-up weekly table with a label every week and the week each label was observed."""
    idx = pd.date_range(start, end, freq="W-SUN")
    t = pd.DataFrame({"target": [i % 3 for i in range(len(idx))]}, index=idx, dtype=float)
    t["target_week"] = idx + pd.Timedelta(weeks=horizon)
    return t


def test_select_labelled_rows_drops_unlabelled_weeks_and_adds_bloom_year():
    t = pd.DataFrame({"target": [0.0, None, 2.0]}, index=pd.to_datetime(["2015-07-26", "2015-08-02", "2015-08-09"]))
    lab = select_labelled_rows(t)
    assert lab["target"].tolist() == [0, 2] and lab["target"].dtype.kind == "i"
    assert lab["bloom_year"].tolist() == [2015, 2016]


def test_forward_folds_start_in_2016_and_train_only_on_earlier_years():
    lab = select_labelled_rows(weekly_table())
    folds = list(make_folds(lab, "forward"))
    assert [y for y, _, _ in folds] == [2016, 2017, 2018]
    assert FORWARD_FIRST_TEST_YEAR == 2016
    for year, tr, te in folds:
        assert (lab["bloom_year"][tr] < year).all()
        assert (lab["bloom_year"][te] == year).all()


def test_loyo_folds_cover_every_year_and_train_on_the_others():
    lab = select_labelled_rows(weekly_table())
    folds = list(make_folds(lab, "loyo"))
    assert [y for y, _, _ in folds] == sorted(lab["bloom_year"].unique())
    for year, tr, te in folds:
        assert not (tr & te).any()
        assert (lab["bloom_year"][tr] != year).all()


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_purging_removes_the_last_h_training_weeks_before_the_test_year(horizon):
    lab = select_labelled_rows(weekly_table(horizon=horizon))
    year, tr, _ = next(make_folds(lab, "forward"))
    earlier = (lab["bloom_year"] < year).to_numpy()
    purged = lab.index[earlier & ~tr]
    assert len(purged) == horizon
    assert (purged == lab.index[earlier][-horizon:]).all()


def test_no_purging_without_target_week():
    lab = select_labelled_rows(weekly_table().drop(columns="target_week"))
    year, tr, _ = next(make_folds(lab, "forward"))
    assert tr.sum() == (lab["bloom_year"] < year).sum()


def test_make_folds_rejects_unknown_schemes():
    with pytest.raises(ValueError, match="unknown scheme"):
        list(make_folds(select_labelled_rows(weekly_table()), "random"))
