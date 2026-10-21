"""Load and clean each project dataset, logging how many rows survive each step.

Every loader returns a tidy DataFrame and appends its steps to a CleaningLog so
the data-review notebook can show the "start at 100%, end at x%" table the
Challenge Advisor asked for. Cleaning decisions and the reason for each are in
the comments next to the step.

Datasets (all from the 1A repo data/ folder):
  1 habsos_cellcounts            K. brevis cell counts      -> LABEL + lag features
  2 ChlorophyllaConcentration*   satellite chl-a (2 files)  -> via processed/chlorophyll_merged.parquet
  3 NutrientsFL_2006_2025        Sarasota Bay + creek water quality
  4 bradentonFL_tempData         daily air temp + rainfall
  5 Sarasota_Wind&Temp           KSRQ airport hourly weather (2022-07+)
  6 SarasotaWindData             gridded zonal wind u (2019+)
  7 MoteMarine_BottomTemp        hourly water temp at Mote (2020-08 to 2025-03)
"""
import numpy as np
import pandas as pd

from .cleaning_log import CleaningLog
from .cleaning_rules import NUTRIENT_MAP, NUTRIENT_VALID
from .config import DATA_DIR, REGIONAL_BOX, TARGET_BOX
from .helpers import is_in_box, read_erddap


# ---------------------------------------------------------------------------
# 1. HABSOS K. brevis cell counts
# ---------------------------------------------------------------------------
def load_habsos(log: CleaningLog) -> pd.DataFrame:
    """Load and clean the HABSOS K. brevis cell counts (the label source).

    Steps: valid date and count; provider QA flag = 1; drop duplicate samples;
    surface samples (≤ 1 m or depth not recorded); regional box. Out-of-range
    salinity and water temperature are blanked, not dropped.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per sample in the regional box, with columns
        date, latitude, longitude, description, cellcount (cells/L), salinity,
        water_temp (°C) and in_target (True inside the Sarasota target box).

    Example:
        >>> h = load_habsos(CleaningLog())
        >>> len(h), int(h["in_target"].sum())
        (114545, 53193)
    """
    name = "habsos_cellcounts"
    df = pd.read_csv(DATA_DIR / "habsos_cellcounts.csv", low_memory=False)
    log.record_step(name, "raw", df)

    df["date"] = pd.to_datetime(df["SAMPLE_DATE"], errors="coerce")
    df = df.dropna(subset=["date", "CELLCOUNT"])
    log.record_step(name, "valid date + cell count", df)

    # CELLCOUNT_QA (HABSOS data dictionary): 1 = good, 4 = exceeds normals, 5 = converted by script, 9 = no data.
    # Only 1 and 5 occur; the 192 flag-5 rows are all Texas 2023, so dropping them doesn't touch Sarasota.
    df = df[df["CELLCOUNT_QA"] == 1]
    log.record_step(name, "cell-count QA flag == 1", df)

    # Same site/date/depth/count entered twice is a re-entry, not a new sample.
    df = df.drop_duplicates(subset=["LATITUDE", "LONGITUDE", "SAMPLE_DATE", "SAMPLE_DEPTH", "CELLCOUNT"])
    log.record_step(name, "drop duplicate site/date/depth/count", df)

    # Surface samples only: FWRI's protocol samples at ~0.5 m (also the median depth here).
    # Null depth = near-surface per the HABSOS data dictionary, so those rows are kept.
    df = df[df["SAMPLE_DEPTH"].isna() | (df["SAMPLE_DEPTH"] <= 1)]
    log.record_step(name, "surface samples (depth <= 1 m)", df)

    df = df[is_in_box(df["LATITUDE"], df["LONGITUDE"], REGIONAL_BOX)].copy()
    log.record_step(name, "inside regional box", df, "label uses the smaller target box")

    # NOTE: we deliberately do NOT drop rows with missing salinity/water temp.
    # The team's earlier cleaned file did, which removed ~44% of all rows - and
    # with them the cell counts that are our label.
    df.loc[~df["SALINITY"].between(0, 45), "SALINITY"] = np.nan
    df.loc[~df["WATER_TEMP"].between(5, 40), "WATER_TEMP"] = np.nan

    df["in_target"] = is_in_box(df["LATITUDE"], df["LONGITUDE"], TARGET_BOX)
    log.record_step(name, "of which inside target box", int(df["in_target"].sum()))
    return df[["date", "LATITUDE", "LONGITUDE", "DESCRIPTION", "CELLCOUNT",
               "SALINITY", "WATER_TEMP", "in_target"]].rename(columns=str.lower)


# ---------------------------------------------------------------------------
# 2. Satellite chlorophyll-a (merged by merge_chlorophyll.ipynb)
# ---------------------------------------------------------------------------
def load_chlorophyll(log: CleaningLog) -> pd.DataFrame:
    """Load the merged satellite chlorophyll-a and keep the target box.

    Reads data/processed/chlorophyll_merged.parquet, written by
    notebooks/merge_chlorophyll.ipynb (file 1 plus quantile-mapped file 2).

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per valid pixel-day in the target box, with columns
        date, latitude, longitude, chlor_a (mg/m³) and coastal (1 within 10 km of land).

    Raises:
        FileNotFoundError: If the merged file has not been created yet.

    Example:
        >>> len(load_chlorophyll(CleaningLog()))
        502775
    """
    name = "chlorophyll (satellite)"
    path = DATA_DIR / "processed" / "chlorophyll_merged.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run notebooks/merge_chlorophyll.ipynb first.")
    df = pd.read_parquet(path)
    log.record_step(name, "merged ds1 + adjusted ds2 (gap-filled 2 km pixels)", df)
    df["date"] = pd.to_datetime(df["date"])
    df = df[is_in_box(df["latitude"], df["longitude"], TARGET_BOX)]
    log.record_step(name, "inside target box", df)
    df = df.dropna(subset=["chlor_a"])
    log.record_step(name, "non-null chl-a", df)
    return df[["date", "latitude", "longitude", "chlor_a", "coastal"]]


# ---------------------------------------------------------------------------
# 3. Sarasota water-quality monitoring (estuary + coastal creeks)
# ---------------------------------------------------------------------------
def load_nutrients(log: CleaningLog) -> pd.DataFrame:
    """Load and clean the Sarasota County water-quality measurements.

    Steps: drop exact duplicates; standardise names via NUTRIENT_MAP (cleaning_rules.py); set
    not-detected (U) values to half the detection limit; remove values outside
    NUTRIENT_VALID (cleaning_rules.py); drop near-bottom readings; label each row bay or creek.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: Long format, one row per measurement, with columns
        date, source ("bay" or "creek"), var (short name, e.g. "tn") and value.

    Example:
        >>> n = load_nutrients(CleaningLog())
        >>> len(n), sorted(n["source"].unique())
        (447524, ['bay', 'creek'])
    """
    name = "NutrientsFL_2006_2025"
    df = pd.read_csv(DATA_DIR / "NutrientsFL_2006_2025.csv", low_memory=False)
    log.record_step(name, "raw", df)
    # Identical rows (same station, time, characteristic and value) are re-entries, not new measurements.
    df = df.drop_duplicates()
    log.record_step(name, "drop exact duplicate rows", df)
    df["date"] = pd.to_datetime(df["Activity_Start_Date"], errors="coerce")
    df["var"] = df["Characteristic"].str.lower().map(NUTRIENT_MAP)
    df = df.dropna(subset=["date", "var", "Result_Value"])
    log.record_step(name, "all characteristics, names standardised", df)

    # 'U' = analyte not detected; reported value is the detection limit.
    # Half the detection limit is the standard substitution for censored values.
    nd = df["Value_Qualifier"].astype(str).str.contains("U")
    df.loc[nd, "Result_Value"] = df.loc[nd, "Result_Value"] / 2

    keep = np.ones(len(df), bool)
    for var, (lo, hi) in NUTRIENT_VALID.items():
        m = (df["var"] == var).to_numpy()
        keep &= ~(m & ~df["Result_Value"].between(lo, hi).to_numpy())
    df = df[keep]
    log.record_step(name, "values outside the expected range removed", df)

    # Surface + midwater only; near-bottom readings describe a different layer
    # than the surface water where K. brevis is sampled.
    df = df[df["RelativeDepth"] != "Near Bottom"]
    log.record_step(name, "drop near-bottom readings", df)
    df["source"] = df["DataSource"].map({"SARASOTAES_WQ": "bay", "SARASOTA_COASTALCREEK_WQ": "creek"})
    return df[["date", "source", "var", "Result_Value"]].rename(columns={"Result_Value": "value"})


# ---------------------------------------------------------------------------
# 4. Bradenton daily air temperature + rainfall
# ---------------------------------------------------------------------------
def load_bradenton(log: CleaningLog) -> pd.DataFrame:
    """Load the Bradenton daily air temperature and rain record.

    −99.9 / −99.99 missing-data codes become NaN, and blank mean temperatures
    are filled with (max + min) / 2, the standard daily-mean definition.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per day, with columns date, precipitation (inches)
        and max_temp, min_temp, mean_temp (°F).

    Example:
        >>> b = load_bradenton(CleaningLog())
        >>> len(b), str(b["date"].min().date()), str(b["date"].max().date())
        (4018, '2015-01-01', '2025-12-31')
    """
    name = "bradentonFL_tempData"
    df = pd.read_csv(DATA_DIR / "bradentonFL_tempData.csv", skipinitialspace=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.record_step(name, "raw", df)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    for c in ["precipitation", "max_temp", "min_temp", "mean_temp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -99, c] = np.nan   # -99.9 / -99.99 are documented missing-value sentinels
    # MEAN TEMP is blank for 18% of days; the daily mean is (max+min)/2 by definition.
    df["mean_temp"] = df["mean_temp"].fillna((df["max_temp"] + df["min_temp"]) / 2)
    log.record_step(name, "sentinels -> NaN, mean temp filled from max/min", df)
    return df[["date", "precipitation", "max_temp", "min_temp", "mean_temp"]]


# ---------------------------------------------------------------------------
# 5. KSRQ airport hourly weather
# ---------------------------------------------------------------------------
def load_ksrq(log: CleaningLog) -> pd.DataFrame:
    """Load the hourly Sarasota–Bradenton airport (KSRQ) weather reports.

    Wind direction and speed are split into east–west (u) and north–south (v)
    parts so they can be averaged; out-of-range air and dew-point temperatures are
    blanked; reports with neither wind nor air temperature are dropped.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per report, with columns date, wind_speed, wind_u,
        wind_v (m/s; wind_v < 0 = wind from the north), air_temperature,
        dew_point_temperature (°C) and air_pressure_at_mean_sea_level.

    Example:
        >>> len(load_ksrq(CleaningLog()))
        41178
    """
    name = "Sarasota_Wind&Temp (KSRQ)"
    df = read_erddap(DATA_DIR / "Sarasota_Wind&Temp.csv")
    log.record_step(name, "raw", df)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    # wind_from_direction is where wind blows FROM (met convention). Convert to
    # u (eastward) / v (northward) components so weekly averaging is a vector mean
    # (Li et al. 2021). A northerly, upwelling-favourable wind has v < 0.
    rad = np.deg2rad(df["wind_from_direction"])
    df["wind_u"] = -df["wind_speed"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(rad)
    # Sensor faults: e.g. -60.6 C on 13-14 May 2024 and 50 C on 30 May 2023. Blank the value, keep the wind.
    df.loc[~df["air_temperature"].between(-5, 45), "air_temperature"] = np.nan
    df.loc[~df["dew_point_temperature"].between(-10, 35), "dew_point_temperature"] = np.nan
    df = df.dropna(subset=["wind_speed", "air_temperature"], how="all")
    log.record_step(name, "rows with wind or air temp", df,
             "dropped gust (87% missing) and visibility (not a HAB driver)")
    return df[["date", "wind_speed", "wind_u", "wind_v", "air_temperature",
               "dew_point_temperature", "air_pressure_at_mean_sea_level"]]


# ---------------------------------------------------------------------------
# 6. Gridded zonal wind (u only)
# ---------------------------------------------------------------------------
def load_uwind(log: CleaningLog) -> pd.DataFrame:
    """Load the gridded east–west wind and average it over the regional box.

    Longitude is converted from 0–360° to −180–180° before the box is applied.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per time step (four a day), with columns date and
        uwnd (m/s; negative = wind blowing from the east).

    Example:
        >>> len(load_uwind(CleaningLog()))
        11220
    """
    name = "SarasotaWindData (u grid)"
    df = read_erddap(DATA_DIR / "SarasotaWindData.csv")
    log.record_step(name, "raw", df)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["longitude"] = df["longitude"] - 360        # 0-360 E -> -180..180
    df = df[is_in_box(df["latitude"], df["longitude"], REGIONAL_BOX)].dropna(subset=["uwnd"])
    log.record_step(name, "inside regional box, non-null", df,
             "file has only the east-west (u) component, no v")
    return df.groupby("date", as_index=False)["uwnd"].mean()


# ---------------------------------------------------------------------------
# 7. Mote Marine hourly water temperature
# ---------------------------------------------------------------------------
def load_mote(log: CleaningLog) -> pd.DataFrame:
    """Load the hourly Mote Marine Laboratory water temperature (2 m depth).

    Only readings that passed quality control (QARTOD flag 1) are kept.

    Args:
        log (CleaningLog): Log that receives the row count after each step.

    Returns:
        pandas.DataFrame: One row per hour, with columns date and
        sea_water_temperature (°C).

    Example:
        >>> len(load_mote(CleaningLog()))
        31629
    """
    name = "MoteMarine_BottomTemp"
    df = read_erddap(DATA_DIR / "MoteMarine_BottomTemp.csv")
    log.record_step(name, "raw", df)
    # IOOS QARTOD flags: 1 pass, 3 suspect, 4 fail. Keep passes only (loses 100 rows).
    df = df[df["sea_water_temperature_qc_agg"] == 1]
    log.record_step(name, "QC flag == 1 (pass)", df)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df[["date", "sea_water_temperature"]]
