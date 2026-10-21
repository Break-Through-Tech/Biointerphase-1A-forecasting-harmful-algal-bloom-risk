"""Load and clean each project dataset, logging how many rows survive each step.

Every loader returns a tidy DataFrame and appends its steps to a CleaningLog so
the data-review notebook can show the "start at 100%, end at x%" table the
Challenge Advisor asked for. Cleaning decisions and the reason for each are in
the comments next to the step.

Datasets (all from the 1A repo data/ folder):
  1 habsos_cellcounts            K. brevis cell counts      -> LABEL + lag features
  2 ChlorophyllaConcentration*   satellite chl-a (2 files)  -> via processed/chlorophyll_merged.parquet
  3 NutrientsFL_2006_2025        Sarasota Bay + creek water quality
"""
import numpy as np
import pandas as pd

from .cleaning_log import CleaningLog
from .cleaning_rules import NUTRIENT_MAP, NUTRIENT_VALID
from .config import DATA_DIR, REGIONAL_BOX, TARGET_BOX
from .helpers import is_in_box


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
