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


# ---------------------------------------------------------------- chlorophyll
def test_load_chlorophyll_keeps_target_box_and_valid_values(data_dir):
    (data_dir / "processed").mkdir()
    pd.DataFrame({"date": ["2020-07-01"] * 3, "latitude": [27.3, 27.3, 28.5],
                  "longitude": [-82.6, -82.6, -82.6], "chlor_a": [0.5, np.nan, 2.0],
                  "coastal": [1, 0, 0]}).to_parquet(data_dir / "processed" / "chlorophyll_merged.parquet")
    log = CleaningLog()
    c = loaders.load_chlorophyll(log)
    assert len(c) == 1 and c["chlor_a"].tolist() == [0.5]
    assert steps(log)["inside target box"] == 2


def test_load_chlorophyll_says_which_notebook_to_run_if_missing(data_dir):
    with pytest.raises(FileNotFoundError, match="merge_chlorophyll.ipynb"):
        loaders.load_chlorophyll(CleaningLog())


# ---------------------------------------------------------------- nutrients
def write_nutrients(d):
    rows = [  # source, date, characteristic, value, qualifier, relative depth
        ("SARASOTAES_WQ", "1/5/2015", "Nitrogen", 0.4, "", "Surface"),
        ("SARASOTAES_WQ", "1/5/2015", "Nitrogen", 0.4, "", "Surface"),                 # exact duplicate
        ("SARASOTAES_WQ", "1/5/2015", "Phosphorus as P", 0.02, "U", "Surface"),        # not detected
        ("SARASOTAES_WQ", "1/5/2015", "Temperature, water", 150, "", "Surface"),       # out of range
        ("SARASOTAES_WQ", "1/5/2015", "Nitrogen", 0.5, "", "Near Bottom"),             # near bottom
        ("SARASOTA_COASTALCREEK_WQ", "1/6/2015", "Nitrogen", 1.2, "", "Surface"),
        ("SARASOTAES_WQ", "1/5/2015", "Some other measurement", 3.0, "", "Surface"),  # not in NUTRIENT_MAP
    ]
    pd.DataFrame(rows, columns=["DataSource", "Activity_Start_Date", "Characteristic", "Result_Value",
                                "Value_Qualifier", "RelativeDepth"]).to_csv(d / "NutrientsFL_2006_2025.csv", index=False)


def test_load_nutrients_applies_each_cleaning_step(data_dir):
    write_nutrients(data_dir)
    log = CleaningLog()
    n = loaders.load_nutrients(log)
    s = steps(log)
    assert s["raw"] == 7
    assert s["drop exact duplicate rows"] == 6
    assert s["all characteristics, names standardised"] == 5
    assert s["values outside the expected range removed"] == 4
    assert s["drop near-bottom readings"] == 3
    assert len(n) == 3


def test_load_nutrients_halves_not_detected_values(data_dir):
    write_nutrients(data_dir)
    n = loaders.load_nutrients(CleaningLog())
    assert n.loc[n["var"] == "tp", "value"].tolist() == [0.01]


def test_load_nutrients_labels_bay_and_creek(data_dir):
    write_nutrients(data_dir)
    n = loaders.load_nutrients(CleaningLog())
    assert sorted(n["source"].unique()) == ["bay", "creek"]
    assert n.columns.tolist() == ["date", "source", "var", "value"]


# ---------------------------------------------------------------- Bradenton
def test_load_bradenton_blanks_sentinels_and_fills_mean_temperature(data_dir):
    (data_dir / "bradentonFL_tempData.csv").write_text(
        "COOPID, YEAR, MONTH, DAY, PRECIPITATION, MAX TEMP, MIN TEMP, MEAN TEMP\n"
        "80945,2015,1,1,0.5,80,60,70\n"
        "80945,2015,1,2,-99.99,82,62, \n"
        "80945,2015,1,3,0.0,-99.9,60,65\n")
    b = loaders.load_bradenton(CleaningLog()).set_index("date")
    assert np.isnan(b.loc["2015-01-02", "precipitation"])
    assert b.loc["2015-01-02", "mean_temp"] == 72.0          # (82 + 62) / 2
    assert np.isnan(b.loc["2015-01-03", "max_temp"])
    assert b.loc["2015-01-03", "mean_temp"] == 65.0          # recorded value kept


# ---------------------------------------------------------------- airport (KSRQ)
def test_load_ksrq_splits_wind_and_blanks_bad_temperatures(data_dir):
    (data_dir / "Sarasota_Wind&Temp.csv").write_text(
        "time,wind_from_direction,wind_speed,air_temperature,dew_point_temperature,air_pressure_at_mean_sea_level\n"
        "UTC,degrees,m.s-1,degree_Celsius,degree_Celsius,millibars\n"
        "2023-01-01T00:00:00Z,0,2.0,20,10,1015\n"        # from the north
        "2023-01-01T01:00:00Z,90,2.0,60,10,1015\n"       # from the east; air temp 60 invalid
        "2023-01-01T02:00:00Z,,,,,\n")                    # nothing recorded (dropped)
    k = loaders.load_ksrq(CleaningLog())
    assert len(k) == 2
    assert k["wind_u"].round(6).tolist() == [0.0, -2.0]
    assert k["wind_v"].round(6).tolist() == [-2.0, 0.0]
    assert np.isnan(k["air_temperature"].iloc[1])


# ---------------------------------------------------------------- gridded wind
def test_load_uwind_converts_longitude_and_averages_the_box(data_dir):
    (data_dir / "SarasotaWindData.csv").write_text(
        "time,latitude,longitude,uwnd\nUTC,degrees_north,degrees_east,m s-1\n"
        "2019-01-01T00:00:00Z,27.0,277.0,-2.0\n"         # -83.0: inside the regional box
        "2019-01-01T00:00:00Z,27.0,277.5,-4.0\n"         # -82.5: inside
        "2019-01-01T00:00:00Z,27.0,270.0,10.0\n")        # -90.0: outside
    u = loaders.load_uwind(CleaningLog())
    assert u["uwnd"].tolist() == [-3.0]
