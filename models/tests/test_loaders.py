"""Unit tests for common/loaders.py, using small made-up files in the raw formats."""
import numpy as np
import pandas as pd
import pytest

from common import loaders
from common.cleaning_log import CleaningLog


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the loaders at an empty temporary data folder."""
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    return tmp_path


def steps(log):
    """Return {step: rows} from a cleaning log."""
    return {r["step"]: r["rows"] for r in log.rows}


# ---------------------------------------------------------------- HABSOS
def write_habsos(d):
    rows = [  # date, lat, lon, depth, count, qa, salinity, temp
        ("2015-09-01", 27.3, -82.6, 0.5, 200000, 1, 35, 28),   # good, target box
        ("2015-09-01", 27.3, -82.6, 0.5, 200000, 1, 35, 28),   # duplicate of the row above
        ("2015-09-02", 27.3, -82.6, None, 5000, 1, 60, 28),    # no depth recorded (kept); salinity 60 invalid
        ("2015-09-03", 27.3, -82.6, 5.0, 1000, 1, 35, 28),     # deeper than 1 m (dropped)
        ("2015-09-04", 27.3, -82.6, 0.5, 1000, 5, 35, 28),     # QA flag 5 (dropped)
        ("2015-09-05", 26.5, -82.0, 0.5, 0, 1, 35, 50),        # regional box only; temp 50 invalid
        ("2015-09-06", 30.0, -87.0, 0.5, 0, 1, 35, 28),        # outside the regional box (dropped)
    ]
    df = pd.DataFrame(rows, columns=["SAMPLE_DATE", "LATITUDE", "LONGITUDE", "SAMPLE_DEPTH", "CELLCOUNT",
                                     "CELLCOUNT_QA", "SALINITY", "WATER_TEMP"])
    df["DESCRIPTION"] = "site"
    df.to_csv(d / "habsos_cellcounts.csv", index=False)


def test_load_habsos_applies_each_cleaning_step(data_dir):
    write_habsos(data_dir)
    log = CleaningLog()
    h = loaders.load_habsos(log)
    s = steps(log)
    assert s["raw"] == 7
    assert s["cell-count QA flag == 1"] == 6
    assert s["drop duplicate site/date/depth/count"] == 5
    assert s["surface samples (depth <= 1 m)"] == 4
    assert s["inside regional box"] == 3
    assert h["in_target"].tolist() == [True, True, False]


def test_load_habsos_blanks_out_of_range_salinity_and_temperature(data_dir):
    write_habsos(data_dir)
    h = loaders.load_habsos(CleaningLog()).set_index("date")
    assert np.isnan(h.loc["2015-09-02", "salinity"])
    assert np.isnan(h.loc["2015-09-05", "water_temp"])


def test_load_habsos_returns_lower_case_columns(data_dir):
    write_habsos(data_dir)
    h = loaders.load_habsos(CleaningLog())
    assert h.columns.tolist() == ["date", "latitude", "longitude", "description", "cellcount",
                                  "salinity", "water_temp", "in_target"]
