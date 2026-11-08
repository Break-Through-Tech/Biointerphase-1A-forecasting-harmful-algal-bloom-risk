"""Unit tests for common/validation.py."""
import pandas as pd

from common.validation import select_labelled_rows


def test_select_labelled_rows_drops_unlabelled_weeks_and_adds_bloom_year():
    t = pd.DataFrame({"target": [0.0, None, 2.0]}, index=pd.to_datetime(["2015-07-26", "2015-08-02", "2015-08-09"]))
    lab = select_labelled_rows(t)
    assert lab["target"].tolist() == [0, 2] and lab["target"].dtype.kind == "i"
    assert lab["bloom_year"].tolist() == [2015, 2016]
