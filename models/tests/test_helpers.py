"""Unit tests for common/helpers.py."""
import numpy as np
import pandas as pd

from common.helpers import assign_class, is_in_box, read_erddap

BOX = {"lat": (26.9, 27.6), "lon": (-83.0, -82.3)}


def test_is_in_box_inside_and_outside():
    lat = pd.Series([27.3, 28.0, 27.3])
    lon = pd.Series([-82.6, -82.6, -84.0])
    assert is_in_box(lat, lon, BOX).tolist() == [True, False, False]


def test_is_in_box_includes_the_edges():
    assert is_in_box(pd.Series([26.9, 27.6]), pd.Series([-83.0, -82.3]), BOX).tolist() == [True, True]


def test_is_in_box_treats_missing_coordinates_as_outside():
    assert is_in_box(pd.Series([np.nan]), pd.Series([-82.6]), BOX).tolist() == [False]


def test_read_erddap_skips_the_units_row(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("time,value\nUTC,m\n2024-01-01T00:00:00Z,1.5\n2024-01-02T00:00:00Z,2.5\n")
    df = read_erddap(f)
    assert len(df) == 2 and df["value"].tolist() == [1.5, 2.5]


def test_read_erddap_passes_options_to_read_csv(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("time,value,other\nUTC,m,-\n2024-01-01T00:00:00Z,1.5,a\n")
    assert read_erddap(f, usecols=["time", "value"]).columns.tolist() == ["time", "value"]


def test_assign_class_thresholds():
    cells = pd.Series([0, 9999, 10000, 99999, 100000, 5e6])
    assert assign_class(cells).tolist() == [0, 0, 1, 1, 2, 2]


def test_assign_class_keeps_missing_as_missing():
    assert np.isnan(assign_class(pd.Series([np.nan])).iloc[0])
