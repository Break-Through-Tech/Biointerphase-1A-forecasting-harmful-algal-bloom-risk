"""Turn the cleaned datasets into one weekly modelling table for the Sarasota coast.

Row t = one week (ending Sunday). Every feature uses only data observed up to
the end of week t; the label is the K. brevis class observed in week t+HORIZON.
That one-way arrow is what makes this a forecast and not a nowcast.
"""
import numpy as np
import pandas as pd

from . import loaders
from .cleaning_log import CleaningLog
from .config import CACHE_DIR, HORIZON_WEEKS, STUDY_START, WEEK_RULE
from .feature_sets import NUTRIENT_EXTRA
from .helpers import assign_class


def build_kb_features(habsos: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the weekly bloom-history features and the observed class.

    Target-box statistics (highest, 95th-percentile and mean count, share of samples
    ≥ 10⁴ and ≥ 10⁵ cells/L, sample count), regional-box statistics, the last observed
    class, weeks since a sample and since a bloom, 1- and 2-week lags, and 4- and
    8-week trailing maxima. Counts use log10(count + 1).

    Args:
        habsos (pandas.DataFrame): Output of loaders.load_habsos.
        weeks (pandas.DatetimeIndex): Week-ending dates to build rows for.

    Returns:
        pandas.DataFrame: One row per week with kb_max (raw), kb_class (observed
        class, NaN if not sampled) and the bloom-history feature columns.

    Example:
        >>> weeks = pd.date_range("2010-01-01", "2026-05-03", freq=WEEK_RULE)
        >>> f = build_kb_features(loaders.load_habsos(CleaningLog()), weeks)
        >>> int(f["kb_class"].notna().sum())
        740
    """
    tgt = habsos[habsos["in_target"]].set_index("date")["cellcount"].resample(WEEK_RULE)
    reg = habsos.set_index("date")["cellcount"].resample(WEEK_RULE)
    f = pd.DataFrame(index=weeks)
    f["kb_n_samples"] = tgt.size().reindex(weeks).fillna(0)
    f["kb_max"] = tgt.max().reindex(weeks)                 # raw, used for the label
    f["kb_max_log"] = np.log10(f["kb_max"] + 1)
    f["kb_p95_log"] = np.log10(tgt.quantile(0.95).reindex(weeks) + 1)
    f["kb_mean_log"] = np.log10(tgt.mean().reindex(weeks) + 1)
    f["kb_frac_1e4"] = tgt.apply(lambda s: (s >= 1e4).mean() if len(s) else np.nan).reindex(weeks)
    f["kb_frac_1e5"] = tgt.apply(lambda s: (s >= 1e5).mean() if len(s) else np.nan).reindex(weeks)
    f["rkb_max_log"] = np.log10(reg.max().reindex(weeks) + 1)
    f["rkb_frac_1e5"] = reg.apply(lambda s: (s >= 1e5).mean() if len(s) else np.nan).reindex(weeks)
    f["rkb_n_samples"] = reg.size().reindex(weeks).fillna(0)

    f["kb_class"] = assign_class(f["kb_max"])                 # observed class this week (NaN = not sampled)
    # Most recent observed class and how stale it is: what a manager "knows" at week t.
    f["last_class"] = f["kb_class"].ffill()
    seen = f["kb_class"].notna()
    f["weeks_since_sample"] = seen.groupby(seen.cumsum()).cumcount()
    f.loc[seen.cumsum() == 0, "weeks_since_sample"] = np.nan
    bloom = (f["kb_class"] == 2)
    f["weeks_since_bloom"] = bloom.groupby(bloom.cumsum()).cumcount().astype(float)
    f.loc[bloom.cumsum() == 0, "weeks_since_bloom"] = np.nan

    for lag in (1, 2):
        for c in ["kb_max_log", "kb_frac_1e5", "rkb_max_log"]:
            f[f"{c}_L{lag}"] = f[c].shift(lag)
    # 4- and 8-week trailing max: has there been a bloom recently even if this week was unsampled?
    f["kb_max_log_4w"] = f["kb_max_log"].rolling(4, min_periods=1).max()
    f["kb_max_log_8w"] = f["kb_max_log"].rolling(8, min_periods=1).max()
    f["rkb_max_log_4w"] = f["rkb_max_log"].rolling(4, min_periods=1).max()
    return f


def build_chlorophyll_features(chl: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the weekly satellite chlorophyll-a features.

    Daily target-box statistics (90th percentile, share of pixels > 1 and > 5 mg/m³,
    coastal mean) are averaged by week; anomalies compare each week with the mean of
    the 8 weeks before the last 2.

    Args:
        chl (pandas.DataFrame): Output of loaders.load_chlorophyll.
        weeks (pandas.DatetimeIndex): Week-ending dates to build rows for.

    Returns:
        pandas.DataFrame: One row per week with chl_p90, chl_frac_gt1, chl_frac_gt5,
        chl_coastal_mean, chl_mean_log, chl_anomaly, chl_coastal_anomaly and
        chl_mean_log_change_1w.

    Example:
        >>> weeks = pd.date_range("2010-01-01", "2026-05-03", freq=WEEK_RULE)
        >>> build_chlorophyll_features(loaders.load_chlorophyll(CleaningLog()), weeks).shape[1]
        8
    """
    daily = chl.groupby("date")["chlor_a"]
    d = pd.DataFrame({
        "chl_mean": daily.mean(),
        "chl_p90": daily.quantile(0.9),
        "chl_frac_gt1": daily.apply(lambda s: (s > 1).mean()),
        "chl_frac_gt5": daily.apply(lambda s: (s > 5).mean()),
    })
    coastal = chl[chl["coastal"] == 1].groupby("date")["chlor_a"].mean().rename("chl_coastal_mean")
    d = d.join(coastal)
    f = d.resample(WEEK_RULE).mean().reindex(weeks)
    f["chl_mean_log"] = np.log10(f["chl_mean"])
    # NOAA operational red-tide flag (Stumpf et al. 2003; Tomlinson et al. 2009):
    # chl-a minus the mean of the previous two months, skipping the latest 2 weeks.
    # An anomaly of ~1 mg/m3 corresponds to roughly 1e4 K. brevis cells/L.
    baseline = f["chl_mean"].shift(2).rolling(8, min_periods=3).mean()
    f["chl_anomaly"] = f["chl_mean"] - baseline
    f["chl_coastal_anomaly"] = f["chl_coastal_mean"] - f["chl_coastal_mean"].shift(2).rolling(8, min_periods=3).mean()
    f["chl_mean_log_change_1w"] = f["chl_mean_log"].diff()
    return f.drop(columns=["chl_mean"])


def build_nutrient_features(nut: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the weekly water-quality features (trailing 8-week medians).

    Each characteristic is reduced to a weekly median per program (bay or creek),
    then to a trailing 8-week median, because samples were roughly monthly before 2016.

    Args:
        nut (pandas.DataFrame): Output of loaders.load_nutrients.
        weeks (pandas.DatetimeIndex): Week-ending dates to build rows for.

    Returns:
        pandas.DataFrame: One row per week with the 11 core columns (e.g. bay_tn_8w),
        the 20 NUTRIENT_EXTRA columns (suffix _8w) and bay_water_temp_c (weekly, °C).

    Example:
        >>> weeks = pd.date_range("2010-01-01", "2026-05-03", freq=WEEK_RULE)
        >>> f = build_nutrient_features(loaders.load_nutrients(CleaningLog()), weeks)
        >>> f.columns[:3].tolist()
        ['bay_tn_8w', 'bay_tp_8w', 'bay_nox_8w']
    """
    wk = (nut.assign(week=nut["date"].dt.to_period(WEEK_RULE).dt.end_time.dt.normalize())
          .pivot_table(index="week", columns=["source", "var"], values="value", aggfunc="median"))
    wk.columns = [f"{s}_{v}" for s, v in wk.columns]
    wk = wk.reindex(weeks)
    # Grab samples are roughly monthly before 2016, so use a trailing 8-week median.
    f = wk.rolling(8, min_periods=1).median()
    keep = ["bay_tn", "bay_tp", "bay_nox", "bay_po4", "bay_chl_insitu", "bay_spcond",
            "bay_turbidity", "bay_do", "creek_tn", "creek_tp", "creek_nox"]
    # Core characteristics first, then every other characteristic (tested later, see feature_sets.py).
    f = pd.concat([f[[c for c in keep if c in f]], f.reindex(columns=NUTRIENT_EXTRA)], axis=1).add_suffix("_8w")
    f["bay_water_temp_c"] = (wk.get("bay_water_temp_f") - 32) * 5 / 9
    return f


def build_weather_features(brad: pd.DataFrame, ksrq: pd.DataFrame, uwind: pd.DataFrame,
                     weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the weekly air-temperature, rain and wind features.

    Args:
        brad (pandas.DataFrame): Output of loaders.load_bradenton.
        ksrq (pandas.DataFrame): Output of loaders.load_ksrq.
        uwind (pandas.DataFrame): Output of loaders.load_uwind.
        weeks (pandas.DatetimeIndex): Week-ending dates to build rows for.

    Returns:
        pandas.DataFrame: One row per week with air_temp_f, air_temp_max_f, rain over
        the last 1, 4, 8 and 12 weeks (inches), airport wind (u, v, speed, 4-week means),
        airport air temperature and pressure, and gridded east–west wind (weekly and
        4-week mean). Airport columns are NaN before July 2022, gridded wind before 2019.

    Example:
        >>> weeks = pd.date_range("2024-01-07", periods=3, freq=WEEK_RULE)
        >>> log = CleaningLog()
        >>> f = build_weather_features(loaders.load_bradenton(log), loaders.load_ksrq(log), loaders.load_uwind(log), weeks)
        >>> "rain_in_4w" in f and "wind_v_4w" in f
        True
    """
    b = brad.set_index("date").resample(WEEK_RULE).agg(
        {"precipitation": "sum", "mean_temp": "mean", "max_temp": "max"}).reindex(weeks)
    f = pd.DataFrame(index=weeks)
    f["air_temp_f"] = b["mean_temp"]
    f["air_temp_max_f"] = b["max_temp"]
    # Rain pulses wash nutrients off land with a lag (Phlips et al. 2020); sum over several windows.
    for w in (1, 4, 8, 12):
        f[f"rain_in_{w}w"] = b["precipitation"].rolling(w, min_periods=max(1, w // 2)).sum()

    k = ksrq.set_index("date").resample(WEEK_RULE).mean().reindex(weeks)
    f["wind_u"] = k["wind_u"]
    f["wind_v"] = k["wind_v"]
    f["wind_speed"] = k["wind_speed"]
    # Li et al. (2021): sustained northerly (v<0, upwelling-favourable) wind raises bloom odds.
    f["wind_v_4w"] = f["wind_v"].rolling(4, min_periods=2).mean()
    f["wind_u_4w"] = f["wind_u"].rolling(4, min_periods=2).mean()
    f["ksrq_air_temp_c"] = k["air_temperature"]
    f["pressure_mb"] = k["air_pressure_at_mean_sea_level"]

    u = uwind.set_index("date")["uwnd"].resample(WEEK_RULE).mean().reindex(weeks)
    f["grid_uwnd"] = u
    f["grid_uwnd_4w"] = u.rolling(4, min_periods=2).mean()
    return f


def build_water_temp_features(habsos: pd.DataFrame, nut_f: pd.DataFrame, mote: pd.DataFrame,
                        weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the weekly water-temperature and salinity features.

    One combined water-temperature series fills each week from the Mote sensor first,
    then the bay water-quality stations, then the HABSOS samples, and interpolates gaps
    of up to 2 weeks.

    Args:
        habsos (pandas.DataFrame): Output of loaders.load_habsos.
        nut_f (pandas.DataFrame): Output of build_nutrient_features (uses bay_water_temp_c).
        mote (pandas.DataFrame): Output of loaders.load_mote.
        weeks (pandas.DatetimeIndex): Week-ending dates to build rows for.

    Returns:
        pandas.DataFrame: One row per week with mote_water_temp_c, habsos_water_temp_c,
        habsos_salinity, water_temp_c (combined, °C), water_temp_change_2w and
        warm_weeks_8w (weeks above 28 °C in the last 8).

    Example:
        >>> weeks = pd.date_range("2010-01-01", "2026-05-03", freq=WEEK_RULE)
        >>> log = CleaningLog()
        >>> nut_f = build_nutrient_features(loaders.load_nutrients(log), weeks)
        >>> f = build_water_temp_features(loaders.load_habsos(log), nut_f, loaders.load_mote(log), weeks)
        >>> float(f["warm_weeks_8w"].max())
        8.0
    """
    f = pd.DataFrame(index=weeks)
    f["mote_water_temp_c"] = mote.set_index("date")["sea_water_temperature"].resample(WEEK_RULE).mean().reindex(weeks)
    tgt = habsos[habsos["in_target"]].set_index("date")
    f["habsos_water_temp_c"] = tgt["water_temp"].resample(WEEK_RULE).mean().reindex(weeks)
    f["habsos_salinity"] = tgt["salinity"].resample(WEEK_RULE).mean().reindex(weeks)
    # One composite series with the longest coverage: Mote sensor > bay WQ > HABSOS sample.
    composite = f["mote_water_temp_c"].fillna(nut_f["bay_water_temp_c"]).fillna(f["habsos_water_temp_c"])
    f["water_temp_c"] = composite.interpolate(limit=2)
    f["water_temp_change_2w"] = f["water_temp_c"].diff(2)
    # Weeks warmer than 28 C in the last 8: a warm-season indicator (28 C is a project choice, not from a cited source).
    f["warm_weeks_8w"] = (f["water_temp_c"] > 28).rolling(8, min_periods=1).sum()
    return f


def build_weekly(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (or load from cache) the weekly modelling table and the cleaning log.

    Runs every loader, builds all feature groups from 2008 (so lags and rolling windows
    have history), trims to STUDY_START, adds the season features and the label
    (the class HORIZON_WEEKS ahead), and caches the result in CACHE_DIR. The last row
    is the last Sunday on or before the latest HABSOS sample, so a week still in progress
    is left out.

    Args:
        rebuild (bool): If True, rebuild from the raw data even if a cached table exists.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: The weekly table (one row per week
        ending Sunday, features plus target and target_week) and the cleaning log.

    Example:
        >>> table, log = build_weekly()
        >>> table.shape[0], int(table["target"].notna().sum())   # with HAB_HORIZON=1
        (853, 739)
    """
    out = CACHE_DIR / f"weekly_sarasota_h{HORIZON_WEEKS}.parquet"
    log_out = CACHE_DIR / "cleaning_log.csv"
    if out.exists() and not rebuild:
        return pd.read_parquet(out), pd.read_csv(log_out)

    log = CleaningLog()
    habsos = loaders.load_habsos(log)
    chl = loaders.load_chlorophyll(log)
    nut = loaders.load_nutrients(log)
    brad = loaders.load_bradenton(log)
    ksrq = loaders.load_ksrq(log)
    uwind = loaders.load_uwind(log)
    mote = loaders.load_mote(log)
    loaders.review_chirps(log)
    loaders.review_cyanotoxins(log)

    end = habsos["date"].max()
    weeks = pd.date_range(STUDY_START, end, freq=WEEK_RULE)
    # Lag/rolling features need history before STUDY_START, so build from 2008 then trim.
    weeks_ext = pd.date_range("2008-01-01", end, freq=WEEK_RULE)

    kb = build_kb_features(habsos, weeks_ext)
    nut_f = build_nutrient_features(nut, weeks_ext)
    table = pd.concat([
        kb,
        build_chlorophyll_features(chl, weeks_ext),
        nut_f.drop(columns=["bay_water_temp_c"]),
        build_water_temp_features(habsos, nut_f, mote, weeks_ext),
        build_weather_features(brad, ksrq, uwind, weeks_ext),
    ], axis=1)
    woy = table.index.isocalendar().week.astype(float).to_numpy()
    table["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    table["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)

    # Label: class observed HORIZON weeks ahead. Weeks with no sampling have no label.
    table["target"] = table["kb_class"].shift(-HORIZON_WEEKS)
    table["target_week"] = table.index + pd.Timedelta(weeks=HORIZON_WEEKS)
    table = table.loc[weeks[0]:]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    log.make_table().to_csv(log_out, index=False)
    return table, log.make_table()
