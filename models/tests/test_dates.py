"""Unit tests for common/dates.py."""
import pandas as pd

from common.dates import assign_bloom_year


def test_july_belongs_to_the_same_years_bloom_year():
    assert assign_bloom_year(pd.DatetimeIndex(["2015-07-31"])).tolist() == [2015]


def test_august_starts_the_next_bloom_year():
    assert assign_bloom_year(pd.DatetimeIndex(["2015-08-01"])).tolist() == [2016]


def test_december_and_january_share_a_bloom_year():
    dates = pd.DatetimeIndex(["2015-12-31", "2016-01-01"])
    assert assign_bloom_year(dates).tolist() == [2016, 2016]


def test_returns_one_value_per_date():
    dates = pd.date_range("2010-01-03", periods=60, freq="W-SUN")
    assert len(assign_bloom_year(dates)) == 60
