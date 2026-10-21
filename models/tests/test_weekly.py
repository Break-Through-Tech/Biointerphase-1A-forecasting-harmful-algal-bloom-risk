"""Unit tests for common/weekly.py, using small made-up tables."""
import numpy as np
import pandas as pd
import pytest

from common import weekly

WEEKS = pd.date_range("2020-01-05", periods=6, freq="W-SUN")   # Sundays 5 Jan – 9 Feb 2020


def habsos_frame():
    """Target-box samples in weeks 1 and 3 (none in weeks 2, 4–6), plus one regional sample."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-15", "2020-01-15"]),
        "cellcount": [200000, 5000, 20000, 1e6],
        "in_target": [True, True, True, False],
        "water_temp": [20.0, 22.0, 21.0, 19.0],
        "salinity": [35.0, 35.0, 36.0, 34.0],
    })


# ---------------------------------------------------------------- bloom history
def test_build_kb_features_weekly_statistics():
    f = weekly.build_kb_features(habsos_frame(), WEEKS)
    assert f.loc["2020-01-05", "kb_max"] == 200000
    assert f.loc["2020-01-05", "kb_n_samples"] == 2
    assert f.loc["2020-01-05", "kb_frac_1e5"] == 0.5
    assert f.loc["2020-01-05", "kb_max_log"] == pytest.approx(np.log10(200001))
    assert f.loc["2020-01-19", "rkb_max_log"] == pytest.approx(np.log10(1e6 + 1))   # regional includes the non-target sample


def test_build_kb_features_class_and_last_class():
    f = weekly.build_kb_features(habsos_frame(), WEEKS)
    assert f["kb_class"].iloc[0] == 2.0 and np.isnan(f["kb_class"].iloc[1]) and f["kb_class"].iloc[2] == 1.0
    assert f["last_class"].tolist()[:4] == [2.0, 2.0, 1.0, 1.0]     # carried forward through unsampled weeks
    assert f["weeks_since_sample"].tolist()[:4] == [0, 1, 0, 1]


def test_build_kb_features_lags_and_rolling_max():
    f = weekly.build_kb_features(habsos_frame(), WEEKS)
    assert f.loc["2020-01-19", "kb_max_log_L2"] == f.loc["2020-01-05", "kb_max_log"]
    assert f.loc["2020-01-26", "kb_max_log_4w"] == f.loc["2020-01-05", "kb_max_log"]   # bloom 3 weeks earlier still counts


# ---------------------------------------------------------------- chlorophyll
def test_build_chlorophyll_features_daily_then_weekly_statistics():
    chl = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 4),
                        "chlor_a": [0.5, 2.0, 6.0, 0.8], "coastal": [1, 1, 0, 0]})
    f = weekly.build_chlorophyll_features(chl, WEEKS)
    row = f.loc["2020-01-05"]
    assert row["chl_frac_gt1"] == 0.5 and row["chl_frac_gt5"] == 0.25
    assert row["chl_coastal_mean"] == pytest.approx(1.25)
    assert row["chl_mean_log"] == pytest.approx(np.log10(np.mean([0.5, 2.0, 6.0, 0.8])))
    assert "chl_mean" not in f


# ---------------------------------------------------------------- water quality
def nut_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-15", "2020-01-01", "2020-01-02"]),
        "source": ["bay", "bay", "creek", "bay"],
        "var": ["tn", "tn", "tn", "water_temp_f"],
        "value": [0.4, 0.6, 1.2, 68.0],
    })


def test_build_nutrient_features_trailing_median_and_columns():
    f = weekly.build_nutrient_features(nut_frame(), WEEKS)
    assert f.loc["2020-01-05", "bay_tn_8w"] == 0.4
    assert f.loc["2020-01-19", "bay_tn_8w"] == 0.5          # median of 0.4 and 0.6 over the trailing 8 weeks
    assert f.loc["2020-01-05", "creek_tn_8w"] == 1.2
    assert {f"{c}_8w" for c in weekly.NUTRIENT_EXTRA} <= set(f.columns)


def test_build_nutrient_features_converts_bay_water_temperature_to_celsius():
    f = weekly.build_nutrient_features(nut_frame(), WEEKS)
    assert f.loc["2020-01-05", "bay_water_temp_c"] == pytest.approx(20.0)


# ---------------------------------------------------------------- weather
def test_build_weather_features_rain_and_wind():
    brad = pd.DataFrame({"date": pd.date_range("2019-12-30", "2020-02-09", freq="D")})
    brad["precipitation"], brad["mean_temp"], brad["max_temp"] = 0.1, 70.0, 80.0
    ksrq = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "wind_u": [-1.0, -3.0],
                         "wind_v": [2.0, 0.0], "wind_speed": [2.0, 3.0], "air_temperature": [20.0, 22.0],
                         "air_pressure_at_mean_sea_level": [1015.0, 1017.0]})
    uwind = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-08"]), "uwnd": [-2.0, -4.0]})
    f = weekly.build_weather_features(brad, ksrq, uwind, WEEKS)
    assert f.loc["2020-01-12", "rain_in_1w"] == pytest.approx(0.7)       # 7 days × 0.1 inch
    assert f.loc["2020-01-05", "wind_u"] == -2.0 and f.loc["2020-01-05", "wind_v"] == 1.0
    assert f.loc["2020-01-12", "grid_uwnd_4w"] == -3.0
    assert np.isnan(f.loc["2020-01-12", "wind_u"])                       # no airport reports that week


# ---------------------------------------------------------------- water temperature
def test_build_water_temp_features_prefers_mote_then_bay_then_habsos():
    nut_f = pd.DataFrame({"bay_water_temp_c": [25.0, 26.0, np.nan, np.nan, np.nan, np.nan]}, index=WEEKS)
    mote = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "sea_water_temperature": [29.0]})
    f = weekly.build_water_temp_features(habsos_frame(), nut_f, mote, WEEKS)
    assert f.loc["2020-01-05", "water_temp_c"] == 29.0        # Mote
    assert f.loc["2020-01-12", "water_temp_c"] == 26.0        # bay water quality
    assert f.loc["2020-01-19", "water_temp_c"] == 21.0        # HABSOS sample
    assert f.loc["2020-01-12", "warm_weeks_8w"] == 1          # one week above 28 C


# ---------------------------------------------------------------- whole table
def test_build_weekly_builds_label_and_uses_the_cache(tmp_path, monkeypatch):
    calls = []
    # one extra no-bloom sample on Mon 20 Jan, so the table runs to the week ending 19 Jan
    h = pd.concat([habsos_frame(), pd.DataFrame({"date": [pd.Timestamp("2020-01-20")], "cellcount": [0],
                                                 "in_target": [True], "water_temp": [20.0], "salinity": [35.0]})],
                  ignore_index=True)
    monkeypatch.setattr(weekly, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(weekly, "STUDY_START", "2020-01-01")
    monkeypatch.setattr(weekly, "HORIZON_WEEKS", 1)
    fake = {
        "load_habsos": lambda log: calls.append("habsos") or h,
        "load_chlorophyll": lambda log: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "chlor_a": [1.0], "coastal": [1]}),
        "load_nutrients": lambda log: nut_frame(),
        "load_bradenton": lambda log: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "precipitation": [0.1],
                                                    "mean_temp": [70.0], "max_temp": [80.0]}),
        "load_ksrq": lambda log: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "wind_u": [0.0], "wind_v": [0.0],
                                               "wind_speed": [0.0], "air_temperature": [20.0],
                                               "air_pressure_at_mean_sea_level": [1015.0]}),
        "load_uwind": lambda log: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "uwnd": [0.0]}),
        "load_mote": lambda log: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "sea_water_temperature": [20.0]}),
        "review_chirps": lambda log: None,
        "review_cyanotoxins": lambda log: None,
    }
    for name, fn in fake.items():
        monkeypatch.setattr(weekly.loaders, name, fn)

    table, log = weekly.build_weekly()
    assert table.index[0] == pd.Timestamp("2020-01-05") and table.index[-1] == pd.Timestamp("2020-01-19")
    assert np.isnan(table.loc["2020-01-05", "target"])        # week ending 12 Jan was not sampled: no label
    assert table.loc["2020-01-12", "target"] == 1.0           # at risk, observed in the week ending 19 Jan
    assert table.loc["2020-01-05", "target_week"] == pd.Timestamp("2020-01-12")
    assert {"woy_sin", "woy_cos"} <= set(table.columns)
    assert (tmp_path / "weekly_sarasota_h1.parquet").exists()

    weekly.build_weekly()                                     # second call reads the cache
    assert calls == ["habsos"]
