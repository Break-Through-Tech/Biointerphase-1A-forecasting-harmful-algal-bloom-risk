"""Generate notebooks/00_master_eda_and_merge.ipynb — a self-contained Colab notebook.

One section per project dataset (what it is, profiling checks, EDA plots, findings,
cleaning decision), then a final section that merges everything into the weekly
modelling table. It does not import `common/`, so it runs on its own in Google Colab.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

# ---------------------------------------------------------------------------
C.append(md("""# Master notebook — EDA of every dataset, then one merged table

**Biointerphase 1A · Forecasting Harmful Algal Bloom (HAB) risk · Break Through Tech AI Studio, Fall 2026**

This notebook goes through the nine project datasets one at a time — what each one is, profiling checks (rows, coverage, missing values, duplicates, invalid values), a few plots, and the cleaning decision — and ends by merging them into the **weekly modelling table** the models are trained on.

| Filename | Website/Source | Dates | Data Included |
|---|---|---|---|
| `habsos_cellcounts` | https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system | 1953 – May 2026 | Latitude, Longitude, Date, Category, Cell Count, Salinity, Water Temp (*K. brevis* only) |
| `ChlorophyllaConcentrationSarasota` + `_Dataset2` | https://coastwatch.noaa.gov/cwn/products/noaa-msl12-multi-sensor-dineof-global-2-km-gap-filled-products-chlorophyll-diffuse.html | 2010 – Sep 2026 (file 1: Jun 2020 – Sep 2026; file 2: Jan 2010 – May 2022) | Latitude, Longitude, Date, Chlorophyll-a (lat 26.5–28, lon −83.5 to −81.8); file 2 is a coarser NOAA CoastWatch product whose exact source isn't documented |
| `NutrientsFL_2006_2025` | https://sarasota.wateratlas.usf.edu/water-quality-trends/ | 1998 – Dec 2025 | Date, Time, Depth; BOD, Chlorophyll-a (pheophytin corrected), DO, DO saturation, Fecal Coliform, Total Nitrogen, Ammonia (N), Kjeldahl Nitrogen, NO₂+NO₃ (N), pH, Total Phosphorus, Phosphate (PO₄), Specific Conductance, Water Temperature, TSS, Turbidity |
| `bradentonFL_tempData` | https://climatecenter.fsu.edu/climate-data-access-tools/downloadable-data | 2015 – 2025 | Date, Precipitation, Max Air Temp, Min Air Temp, Mean Air Temp |
| `Sarasota_Wind&Temp` | https://erddap.secoora.org/erddap/tabledap/gov_noaa_awc_ksrq.html | Jul 2022 – Jun 2026 | Air pressure, air temp, dew point temp, visibility, wind gust speed, wind speed, wind direction |
| `SarasotaWindData` | https://coastwatch.pfeg.noaa.gov/erddap/griddap/pifscCcmpDailyV21NRT.html (CCMP v2.1 NRT) | Jan 2019 – Sep 2026 | Latitude, Longitude, Date, East-west wind (u) |
| `MoteMarine_BottomTemp` | https://erddap.secoora.org/erddap/tabledap/org_secoora_scan_rng_sw.html | Aug 2020 – Mar 2025 | Latitude, Longitude, Date, Depth, Water Temp |
| `chirps20GlobalDailyP05_0878_d358_0bc6` | https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalDailyP05.html | 2016 – 2026 (values only in 2016, 2020, 2024) | Latitude, Longitude, Date, Precipitation (land only) |
| `florida_cyanotoxins_2015-01-01_to_2026-06-17` | https://storymaps.arcgis.com/stories/ccc4c98425194605924ae474fb8002ba | 2015 – 2026 | Latitude, Longitude, Location Type, Date, Depth of Measurement, Cyanotoxin Type, Cyanotoxin Concentration, Detection Limit |"""))

C.append(md("""## 0. Setup"""))

C.append(code("""import os, sys, subprocess
from pathlib import Path

DATA_SOURCE = os.environ.get("HAB_DATA_SOURCE", "drive")      # "drive" | "github" | "local"
DRIVE_DATA_DIR = "/content/drive/MyDrive/BTT Fall AI Studio 2026/Google Colab/data"
GITHUB_REPO = "https://github.com/Break-Through-Tech/Biointerphase-1A-forecasting-harmful-algal-bloom-risk"
LOCAL_DATA_DIR = os.environ.get("HAB_DATA_DIR", str(Path("..") / ".." / "data"))  # the repo's data/ folder

IN_COLAB = "google.colab" in sys.modules

if DATA_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = Path(DRIVE_DATA_DIR)
    OUTPUT_DIR = DATA_DIR.parent / "outputs"
elif DATA_SOURCE == "github":
    repo_dir = Path("/content" if IN_COLAB else ".") / "hab_repo"
    if not repo_dir.exists():
        # Clone without the large files, then pull only the CSVs (skips ~750 MB of duplicate JSON).
        subprocess.run("git lfs install", shell=True, check=True)
        subprocess.run(f"git clone --depth 1 {GITHUB_REPO} {repo_dir}", shell=True, check=True,
                       env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        subprocess.run('git lfs pull --include "data/*.csv"', shell=True, check=True, cwd=repo_dir)
    DATA_DIR = repo_dir / "data"
    OUTPUT_DIR = repo_dir / "outputs"
else:
    DATA_DIR = Path(LOCAL_DATA_DIR)
    OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("data from:", DATA_DIR)
print("outputs to:", OUTPUT_DIR)
print("files:", sorted(p.name for p in DATA_DIR.iterdir() if p.suffix in (".csv", ".txt")))"""))

C.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)   # show cleaning-log notes in full
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.92"})

# --- Study design (shared by every section) --------------------------------
# Target box: Sarasota County coast, Anna Maria Island (27.5 N) to Englewood (26.9 N), ~45 km offshore.
TARGET_BOX = dict(lat=(26.9, 27.6), lon=(-83.0, -82.3))
# Wider box for "what is happening nearby" (blooms drift in from the south/offshore; Medina et al. 2024).
REGIONAL_BOX = dict(lat=(26.3, 28.0), lon=(-83.5, -81.8))
CLASS_EDGES = [0, 1e4, 1e5]                    # no bloom | at risk | bloom
CLASS_NAMES = ["no_bloom", "at_risk", "bloom"]
WEEK_RULE = "W-SUN"                            # weeks end on Sunday
STUDY_START = "2010-01-01"                     # satellite chlorophyll begins Jan 2010
HORIZON_WEEKS = 1                              # label = class in week t+1


def find_file(stem: str) -> Path:
    \"\"\"Find a project data file by name, whatever its extension.

    The same file ends in .csv in the GitHub repo but sometimes .txt on Drive.

    Args:
        stem (str): File name without extension, e.g. "habsos_cellcounts".

    Returns:
        pathlib.Path: Path to the first matching file in DATA_DIR.

    Raises:
        FileNotFoundError: If neither stem.csv nor stem.txt exists in DATA_DIR.

    Example:
        >>> find_file("habsos_cellcounts").name
        'habsos_cellcounts.csv'
    \"\"\"
    for ext in (".csv", ".txt"):
        p = DATA_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}.csv/.txt not found in {DATA_DIR}")


def read_erddap(stem: str, **kw) -> pd.DataFrame:
    \"\"\"Read an ERDDAP CSV export, skipping its units row.

    ERDDAP puts a row of units (e.g. "UTC", "degrees_north") directly under the
    header; read as data it would turn every column into text.

    Args:
        stem (str): File name without extension.
        **kw: Extra keyword arguments passed to pandas.read_csv (e.g. usecols, dtype).

    Returns:
        pandas.DataFrame: The file's rows with the units row removed.

    Example:
        >>> read_erddap("MoteMarine_BottomTemp").columns[0]
        'time'
    \"\"\"
    return pd.read_csv(find_file(stem), skiprows=[1], **kw)


def is_in_box(lat: pd.Series, lon: pd.Series, box: dict) -> pd.Series:
    \"\"\"Flag rows whose coordinates fall inside a latitude/longitude box.

    Args:
        lat (pandas.Series): Latitudes in degrees north.
        lon (pandas.Series): Longitudes in degrees east, -180..180.
        box (dict): {"lat": (min, max), "lon": (min, max)}, bounds inclusive.

    Returns:
        pandas.Series: Boolean mask, True where the point is inside the box.

    Example:
        >>> is_in_box(pd.Series([27.3, 28.5]), pd.Series([-82.6, -82.6]), TARGET_BOX).tolist()
        [True, False]
    \"\"\"
    return lat.between(*box["lat"]) & lon.between(*box["lon"])


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Summarise every column of a DataFrame for a first data-quality look.

    Args:
        df (pandas.DataFrame): The table to profile.

    Returns:
        pandas.DataFrame: One row per column with its dtype, missing count,
            missing percentage, number of distinct values and one example value.

    Example:
        >>> profile_columns(pd.DataFrame({"a": [1, None, 1]}))[["missing", "distinct"]].values.tolist()
        [[1, 1]]
    \"\"\"
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (100 * df.isna().mean()).round(1),
        "distinct": df.nunique(),
        # First non-null value, so a column that is mostly empty still shows what it holds.
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def plot_per_year(dates: pd.Series, title: str, ax: plt.Axes | None = None, color: str = "tab:blue") -> pd.Series:
    \"\"\"Bar chart of how many rows fall in each calendar year.

    Args:
        dates (pandas.Series): Datetime values, one per row.
        title (str): Chart title.
        ax (matplotlib.axes.Axes, optional): Axes to draw on; a new figure is made if None.
        color (str): Bar colour.

    Returns:
        pandas.Series: Row count per year, indexed by year.

    Example:
        >>> plot_per_year(pd.Series(pd.to_datetime(["2020-01-05", "2020-06-01", "2021-03-02"])), "Rows per year").tolist()
        [2, 1]
    \"\"\"
    counts = dates.dt.year.value_counts().sort_index()
    ax = ax or plt.subplots(figsize=(11, 2.8))[1]
    counts.plot(kind="bar", ax=ax, color=color, width=0.85)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("rows")
    return counts


# Cleaning log: every section records rows kept per step -> the "100% -> x%" table at the end.
LOG = []


def log_step(dataset: str, step: str, n: int | pd.DataFrame, note: str = "") -> None:
    \"\"\"Record how many rows of a dataset remain after a cleaning step.

    Appends to the global LOG list, which becomes the cleaning-log table in §10.

    Args:
        dataset (str): Dataset name, e.g. "habsos_cellcounts".
        step (str): Short description of the step, e.g. "QA flag == 1".
        n (int or sized): Row count, or any object with a length (a DataFrame).
        note (str): Optional reason or caveat shown in the log.

    Returns:
        None

    Example:
        >>> log_step("MoteMarine_BottomTemp", "raw", 31729)
        >>> LOG[-1]["rows"]
        31729
    \"\"\"
    n = n if isinstance(n, (int, np.integer)) else len(n)
    LOG.append(dict(dataset=dataset, step=step, rows=int(n), note=note))


def show_step(dataset: str, before: pd.DataFrame, after: pd.DataFrame, label: str) -> None:
    \"\"\"Print how many rows a cleaning step kept and removed, and record it in the log.

    Args:
        dataset (str): Dataset name used in the cleaning log.
        before (pandas.DataFrame): Table before the step.
        after (pandas.DataFrame): Table after the step.
        label (str): Short description of the step.

    Returns:
        None

    Example:
        >>> show_step("habsos_cellcounts", h2, h3, "drop duplicates")
        drop duplicates: 218,960 -> 217,024 rows  (removed 1,936)
    \"\"\"
    diff = len(before) - len(after)
    change = f"removed {diff:,}" if diff >= 0 else f"added {-diff:,}"
    print(f"{label}: {len(before):,} -> {len(after):,} rows  ({change})")
    log_step(dataset, label, after)


def summarise_cleaning(dataset: str, subset_steps: tuple = ()) -> pd.DataFrame:
    \"\"\"Table of rows kept after each logged cleaning step of one dataset.

    Args:
        dataset (str): Dataset name as used in log_step / show_step.
        subset_steps (iterable of str): Steps that select a subset for reporting
            (e.g. "of which inside target box") rather than remove rows; their
            "removed" value is shown as 0.

    Returns:
        pandas.DataFrame: One row per step with rows kept, rows removed and
            percentage of the first logged step.

    Example:
        >>> summarise_cleaning("MoteMarine_BottomTemp")["rows"].tolist()
        [31729, 31629, 31629]
    \"\"\"
    s = pd.DataFrame([r for r in LOG if r["dataset"] == dataset])
    removed = (-s["rows"].diff()).fillna(0).astype(int)
    removed[s["step"].isin(subset_steps)] = 0
    # A negative change means the step added rows (e.g. re-gridding); say so instead of showing a minus sign.
    s["removed"] = [f"{r:,}" if r >= 0 else f"{-r:,} added" for r in removed]
    s["% of raw"] = (100 * s["rows"] / s["rows"].iloc[0]).round(1)
    return s[["step", "rows", "removed", "% of raw"] + (["note"] if s["note"].astype(bool).any() else [])]"""))


nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
out = Path(__file__).with_name("00_master_eda_and_merge.ipynb")
nbf.write(nb, out)
print("wrote", out)
