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

# ---------------------------------------------------------------------------
# 1. HABSOS
C.append(md("""---
## 1. `habsos_cellcounts` — *K. brevis* cell counts (the label)

| | |
|---|---|
| **What** | Water samples counted under a microscope for *Karenia brevis*, the red-tide dinoflagellate |
| **Source** | NOAA HABSOS / Florida Fish and Wildlife Conservation Commission (FWC) |
| **Coverage** | 1953 – May 2026, Gulf of Mexico (mostly Florida) |
| **Role** | The **label** (weekly maximum count in the Sarasota box) and the bloom-history features |

**This section:** 1.1 load and first look · 1.2 data-quality checks · 1.3 the five HABSOS categories · 1.4 exploring the data · 1.5 findings · 1.6 cleaning, step by step · 1.7 cleaning summary"""))

C.append(md("""### 1.1 Load and first look
One row per water sample: where (latitude, longitude, site description), when, how deep, the cell count, and optional salinity / water temperature / wind readings taken with the sample."""))
C.append(code("""habsos_raw = pd.read_csv(find_file("habsos_cellcounts"), low_memory=False)
habsos_raw["date"] = pd.to_datetime(habsos_raw["SAMPLE_DATE"], errors="coerce")
print(f"{len(habsos_raw):,} rows x {habsos_raw.shape[1]} columns")
habsos_raw.head(3)"""))
C.append(md("""**Column overview** — type, how much is missing, how many distinct values, and an example of each column:"""))
C.append(code("""profile_columns(habsos_raw)"""))

C.append(md("""### 1.2 Data-quality checks
Each check asks one question about the raw file; the answer decides whether a cleaning step is needed in 1.6."""))
C.append(code("""both_missing = habsos_raw["SALINITY"].isna() & habsos_raw["WATER_TEMP"].isna()
dup_key = ["LATITUDE", "LONGITUDE", "SAMPLE_DATE", "SAMPLE_DEPTH", "CELLCOUNT"]
checks = pd.DataFrame([
    ("Which organisms are in the file?", str(habsos_raw.groupby(["GENUS", "SPECIES"]).size().to_dict())),
    ("Which states?", str(habsos_raw["STATE_ID"].value_counts().to_dict())),
    ("Date range", f"{habsos_raw['date'].min().date()} -> {habsos_raw['date'].max().date()}"),
    ("Dates that fail to parse", int(habsos_raw["date"].isna().sum())),
    ("Missing cell counts", int(habsos_raw["CELLCOUNT"].isna().sum())),
    ("Negative cell counts", int((habsos_raw["CELLCOUNT"] < 0).sum())),
    ("Cell-count QA flags (1 = good)", str(habsos_raw["CELLCOUNT_QA"].value_counts().to_dict())),
    ("Duplicate samples (same place, date, depth, count)", int(habsos_raw.duplicated(dup_key).sum())),
    ("Samples deeper than 1 m", int((habsos_raw["SAMPLE_DEPTH"] > 1).sum())),
    ("Samples with no depth recorded", int(habsos_raw["SAMPLE_DEPTH"].isna().sum())),
    ("Salinity outside 0-45", int((habsos_raw["SALINITY"].notna() & ~habsos_raw["SALINITY"].between(0, 45)).sum())),
    ("Water temp outside 5-40 °C (implausible)", int((habsos_raw["WATER_TEMP"].notna() & ~habsos_raw["WATER_TEMP"].between(5, 40)).sum())),
    ("Rows missing BOTH salinity and water temp", f"{both_missing.sum():,} ({both_missing.mean():.1%})"),
    ("...of which also missing the cell count", int(habsos_raw.loc[both_missing, "CELLCOUNT"].isna().sum())),
    ("Rows with no CATEGORY", int(habsos_raw["CATEGORY"].isna().sum())),
], columns=["check", "result"])
checks"""))

C.append(md("""### 1.3 The five HABSOS categories
HABSOS bins every sample: `CELLCOUNT` is the raw count (*K. brevis* cells per litre) and `CATEGORY` is HABSOS's label for that count.

| Category | HABSOS rule (cells/L) | Possible effects, per FWC [1] |
|---|---|---|
| not observed | 0 (nothing detected) | no effects anticipated |
| very low | 1 – 9,999 | possible respiratory irritation; shellfish harvesting closes at ≥ 5,000 cells/L |
| low | 10,000 – 99,999 | respiratory irritation; shellfish harvesting closures; possible fish kills; satellites may detect chlorophyll at the upper end |
| medium | 100,000 – 999,999 | respiratory irritation; shellfish harvesting closures; probable fish kills; satellites detect surface chlorophyll |
| high | 1,000,000 or more | as above, plus water discolouration |

**Source**
1. FWC Fish and Wildlife Research Institute, *Red Tide Current Status*, "Key for Results" table. https://myfwc.com/research/redtide/statewide/"""))
C.append(code("""order = ["not observed", "very low", "low", "medium", "high"]
print("Categories in the file:", habsos_raw["CATEGORY"].dropna().unique())
by_cat = habsos_raw.groupby("CATEGORY")["CELLCOUNT"].describe().reindex(order)
by_cat["count"] = by_cat["count"].astype(int)
by_cat"""))
C.append(md("""**What the table shows**

* **The categories don't overlap.** Each one starts just above where the previous one ends, so every count belongs to exactly one category.
* **Edge values go up a class.** A count of exactly 100,000 is "medium", not "low".
* **The units are cells/L**, as the `CELLCOUNT_UNIT` column says.
* **"high" is very skewed.** Its median is about 2 million cells/L, but its maximum is 7.7 billion. This doesn't affect the label, which only asks whether a count is ≥ 100,000."""))
C.append(code("""no_cat = habsos_raw["CATEGORY"].isna()
print(f"Rows with no CATEGORY: {no_cat.sum()} (all of them still have a CELLCOUNT: {habsos_raw.loc[no_cat, 'CELLCOUNT'].notna().all()})")"""))
C.append(md("""**How this becomes the label.** The forecast predicts three classes, made by merging HABSOS's five at the 10,000 and 100,000 boundaries:

| Forecast class | HABSOS categories | Cells/L | Why the boundary sits there [2] |
|---|---|---|---|
| no bloom | not observed + very low | < 10,000 | at most mild respiratory irritation |
| at risk | low | 10,000 – 99,999 | irritation, shellfish closures, possible fish kills: a bloom is starting |
| bloom | medium + high | ≥ 100,000 | probable fish kills; visible to satellites |

* **The three classes match Medina et al. [1].** Their weekly *K. brevis* forecast for neighbouring Charlotte Harbor predicts the five FWC classes, then merges them into the same three superclasses (Background/Very Low, Low, Medium/High). They note the three are what matter for bloom management, mitigation and ecological effects. Merging also raised their accuracy from 0.50 to 0.73 at one week, mostly by joining Medium and High, which their model often confused.
* **Both boundaries come from that same scheme.** 10,000 and 100,000 cells/L are where Medina et al.'s Low and Medium/High superclasses start [1]. Li et al. [3] also define a bloom as ≥ 100,000 cells/L.
* **"at risk" is the early warning.** It flags a rise past 10,000 cells/L before fish kills become likely.

**Sources**
1. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729. https://doi.org/10.1016/j.hal.2024.102729
2. FWC Fish and Wildlife Research Institute, *Red Tide Current Status*, "Key for Results" table. https://myfwc.com/research/redtide/statewide/
3. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))

C.append(md("""### 1.4 Exploring the data
Three questions: how are cell counts distributed, where are samples taken, and when do blooms happen?"""))
C.append(code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pos = habsos_raw.loc[habsos_raw["CELLCOUNT"] > 0, "CELLCOUNT"]
axes[0].hist(np.log10(pos), bins=60, color="tab:blue")
for x in (4, 5): axes[0].axvline(x, color="0.4", ls="--", lw=1)
axes[0].set_title(f"log10 cell count (non-zero only; {len(pos)/len(habsos_raw):.0%} of rows)")
axes[0].set_xlabel("log10 cells/L  (dashed = 10^4 and 10^5)")

sub = habsos_raw.sample(min(40000, len(habsos_raw)), random_state=0)
axes[1].scatter(sub["LONGITUDE"], sub["LATITUDE"], s=1, c=np.log10(sub["CELLCOUNT"] + 1), cmap="viridis")
for box, col in [(TARGET_BOX, "red"), (REGIONAL_BOX, "orange")]:
    axes[1].add_patch(plt.Rectangle((box["lon"][0], box["lat"][0]), np.diff(box["lon"])[0], np.diff(box["lat"])[0],
                                    fill=False, ec=col, lw=1.5))
axes[1].set_xlim(-88, -79.5); axes[1].set_ylim(24, 31)
axes[1].set_title("Sample locations (red = Sarasota target box, orange = regional)")

fl = habsos_raw[habsos_raw["STATE_ID"] == "FL"]
monthly_bloom = fl.groupby(fl["date"].dt.month)["CELLCOUNT"].apply(lambda s: (s >= 1e5).mean())
axes[2].bar(monthly_bloom.index, 100 * monthly_bloom.values, color="tab:red")
axes[2].set_title("Florida: % of samples at bloom level, by month"); axes[2].set_xlabel("month")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: cell-count distribution.** Counts range from about 100 to 10¹⁰ cells/L, so they are log10-transformed for plotting and modelling. On a normal scale the largest blooms would hide everything else. The x-axis shows log10 cells/L, where each step is ten times more cells (2 = 100, 4 = 10,000, 5 = 100,000). The y-axis shows how many samples fall in each bin. Most samples (79 %) contain no *K. brevis* and are not plotted, since log10(0) is undefined. Of those that do, about a third are at bloom level (≥ 10⁵, right dashed line; the left dashed line marks 10⁴).
* **Middle: sample locations.** The map shows where samples were taken, so it's clear which ones fall near Sarasota. The x-axis is longitude and the y-axis is latitude. Each dot is one sample, from a random 40,000 so the map stays readable. Colour shows log10(count + 1), from purple (no cells) to yellow (high counts); the + 1 lets zero counts be plotted. 97 % of samples are from Florida (the rest from Texas, Alabama and Mississippi). Sampling is densest along the southwest coast, around the regional box (orange) and the Sarasota target box (red).
* **Right: bloom season.** The chart shows when in the year blooms occur, using Florida samples only since they are 97 % of the data. The x-axis is the month (1 = January). The y-axis is the percentage of that month's samples at bloom level (≥ 10⁵ cells/L). Blooms are most common in **September–November** (10–12 % of samples) and least common in April–May (about 2 %). This seasonal cycle is what the seasonal-climatology baseline captures, and why month of year is a candidate feature."""))
C.append(md("""**How often is the Sarasota box sampled, and where?**"""))
C.append(code("""tgt_raw = habsos_raw[is_in_box(habsos_raw["LATITUDE"], habsos_raw["LONGITUDE"], TARGET_BOX)]
plot_per_year(tgt_raw.loc[tgt_raw["date"] >= "2000", "date"], "Samples per year inside the Sarasota target box")
plt.show()
print(tgt_raw["DESCRIPTION"].value_counts().head(10))"""))

C.append(md("""### 1.5 Findings

* **Only one organism.** Every row is *Karenia brevis*, so this file can forecast red tide but not other HAB types.
* **Most samples found no *K. brevis*.** In 79 % of water samples (172,626 of 219,152) none were detected (`CELLCOUNT` = 0). The classes are therefore imbalanced, so models are judged on macro-F1 and bloom recall, not accuracy.
* **Counts span about eight orders of magnitude** (≈ 100 to 7.7 × 10⁹ cells/L), so they are log10-transformed for plotting and modelling (1.4).
* **`CATEGORY` and `CELLCOUNT` agree.** The category ranges don't overlap and edge values go up a class (1.3), so the label can come straight from `CELLCOUNT`.
* **The forecast has three classes:** no bloom (< 10⁴), at risk (10⁴–10⁵) and bloom (≥ 10⁵ cells/L). These are HABSOS's five categories merged as in Medina et al. (1.3).
* **The cell count is never missing.** Salinity and water temperature, however, are *both* missing in ~44 % of rows. An earlier team cleaning rule dropped those rows and lost ~95k valid labels; they are kept here.
* **Sampling is concentrated near Sarasota.** 97 % of samples are from Florida and are densest on the southwest coast. Two Sarasota Bay sites (Bay Dock and New Pass Dock) supply about 10,000 samples.
* **Red tide is seasonal.** In Florida, bloom-level samples are most common in September–November (10–12 % of samples), stay elevated from August to January, and drop to about 2 % in April–May.
* **Sampling in the Sarasota box nearly stops in 2022–2023**, the same years the airport wind data begins (§5)."""))

C.append(md("""### 1.6 Cleaning, step by step
Each step shows what it removes and why, then applies it. Rows are only removed when they are wrong or irrelevant; a few out-of-range values are blanked instead."""))
C.append(code("""name = "habsos_cellcounts"
log_step(name, "raw file", habsos_raw)"""))

C.append(md("""#### Step 1 — Keep rows with a valid date and cell count
A row without a date or a count can't become a label. (Check 1.2 found none, so this is a safety net.)"""))
C.append(code("""h1 = habsos_raw.dropna(subset=["date", "CELLCOUNT"])
show_step(name, habsos_raw, h1, "valid date + cell count")"""))

C.append(md("""#### Step 2 — Keep counts the provider marked as good (QA flag = 1)
`CELLCOUNT_QA` is HABSOS's quality flag for the cell count: 1 = good, 4 = exceeds normals, 5 = value converted by script, 9 = no data [1]. Only two values occur here: 1 (218,960 rows) and 5 (192 rows). All 192 flag-5 rows are Texas samples from 2023, and they include the file's largest count (7.7 × 10⁹ cells/L). Only flag-1 rows are kept. This removes under 0.1 % of the data and no samples near Sarasota.

**Source**
1. NOAA NCEI, *HABSOS Data Dictionary*, NCEI Accession 0120767 (v8.8), Support Documents. https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx"""))
C.append(code("""print(h1["CELLCOUNT_QA"].value_counts().rename("rows"))
h2 = h1[h1["CELLCOUNT_QA"] == 1]
show_step(name, h1, h2, "QA flag == 1")"""))

C.append(md("""#### Step 3 — Remove duplicate samples
The same sample sometimes appears twice (same location, date, depth and count) under two record IDs or site names — a re-entry, not a second measurement. Left in, it would count twice in weekly statistics. Example pairs:"""))
C.append(code("""dups = h2[h2.duplicated(dup_key, keep=False)].sort_values(dup_key)
display(dups[dup_key + ["DESCRIPTION", "OBJECTID"]].head(6))
h3 = h2.drop_duplicates(dup_key)
show_step(name, h2, h3, "drop duplicates")"""))

C.append(md("""#### Step 4 — Keep surface samples (≤ 1 m, or depth not recorded)
*K. brevis* is monitored at the surface, so deeper samples (down to 1,000 m) describe a different water layer from the beach and bay conditions being forecast.

* **Protocol:** FWRI samples about 0.5 m below the surface [1], which is also the median depth in this file.
* **Literature:** Li et al. (2021) use only near-surface counts [2]. Medina, reviewing a HABSOS-based *K. brevis* model, asked whether samples below a certain depth were excluded and whether results applied only "within one meter of the surface" [3].
* **Blank depth:** HABSOS states that a missing depth means a near-surface sample [4], so those rows are kept.

**Sources**
1. FWC Fish and Wildlife Research Institute, *HAB sampling protocol for phytoplankton* (Feb 2019), NCEI Accession 0120767 Support Documents. https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/FWRI-HAB%20sampling%20protocol%20for%20phytoplankton%20(Feb2019).pdf
2. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999
3. Medina, M. (2023, 13 July). Reviewer report on Guirhem, G. L., Baker, L., & Moraga, P., Modeling the spatio-temporal distribution of *Karenia brevis* blooms in the Gulf of Mexico. *F1000Research*, 12, 633. In the "Open Peer Review" panel, Methods comments: "Are the results only relevant for K. brevis near the surface (e.g. within one meter of the surface)?" https://f1000research.com/articles/12-633
4. NOAA NCEI, *HABSOS Data Dictionary*, NCEI Accession 0120767 (v8.8), "Additional statements" (row 31): "If a SAMPLE_DEPTH value is null it is assumed to be collected very near to the surface." https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx"""))
C.append(code("""print("Sample depth (m):", h3["SAMPLE_DEPTH"].describe().round(1).to_dict())
print("Depth not recorded:", h3["SAMPLE_DEPTH"].isna().sum())
h4 = h3[h3["SAMPLE_DEPTH"].isna() | (h3["SAMPLE_DEPTH"] <= 1)]
show_step(name, h3, h4, "surface samples (<= 1 m)")"""))

C.append(md("""#### Step 5 — Keep the region around Sarasota
Samples from Texas, Alabama, the Panhandle and the Keys don't describe Sarasota conditions. Only samples in a **regional box** are kept (lat 26.3–28.0, lon −83.5 to −81.8) — wider than the Sarasota target box, because blooms start offshore and are carried to the coast, so "what's happening nearby" is a useful feature (sources in Step 7)."""))
C.append(code("""print("Rows by state before:", h4["STATE_ID"].value_counts().to_dict())
h5 = h4[is_in_box(h4["LATITUDE"], h4["LONGITUDE"], REGIONAL_BOX)].copy()
show_step(name, h4, h5, "inside regional box")"""))

C.append(md("""#### Step 6 — Blank out-of-range salinity and water-temperature values (rows kept)
Salinity outside 0–45 and water temperature outside 5–40 °C are treated as errors. This is the **gross range test**, a required check for ocean temperature and salinity data [1]: any value outside a minimum–maximum span is flagged. The manual's example span is −2.5–40 °C for temperature and 2–41 for salinity, and it lets each operator adjust the span using local knowledge [1]. Two bounds here differ from the example:

* **Temperature lower bound is 5 °C instead of −2.5 °C.** In this data, the only water temperatures below 5 °C are 297 readings between 0.6 and 4.2 °C; every other reading is 6.0 °C or warmer. All 297 carry HABSOS flag 5, "value converted by script" [2]. A 5 °C bound removes exactly those flagged readings and nothing else. The −2.5 °C example bound would keep them.
* **Salinity lower bound is 0 instead of 2.** 118 samples have salinity between 0 and 2. Using 2 would remove them, while 0 keeps them. The upper bound of 45 removes one value, 308, which HABSOS flags as 4, "exceeds normals" [2].

Only the flagged *value* is set to missing; the row and its cell count stay.

**Sources**
1. U.S. Integrated Ocean Observing System (2020). *Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data*, Version 2.1, Test 4 (Gross Range Test), p. 17. https://doi.org/10.25923/x02m-m555
2. NOAA NCEI, *HABSOS Data Dictionary*, NCEI Accession 0120767 (v8.8): QA flags "1=good 4=exceeds normals 5=value converted by script 9=no data". https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx"""))
C.append(code("""bad_sal = h5["SALINITY"].notna() & ~h5["SALINITY"].between(0, 45)
bad_temp = h5["WATER_TEMP"].notna() & ~h5["WATER_TEMP"].between(5, 40)
print(f"salinity values blanked: {bad_sal.sum()} | water-temp values blanked: {bad_temp.sum()}")
h5.loc[bad_sal, "SALINITY"] = np.nan
h5.loc[bad_temp, "WATER_TEMP"] = np.nan"""))

C.append(md("""#### Step 7 — Flag which samples are inside the Sarasota target box
The **target box** (lat 26.9–27.6, lon −83.0 to −82.3; Anna Maria Island to Englewood) defines the label. Samples elsewhere in the regional box (Step 5) stay in the table as "nearby" features.

* **Same design as Medina et al. [1].** They forecast inside a small target domain (Charlotte Harbor), using *K. brevis* features from a larger feature domain that contains it and extends north to Sarasota Bay: "The K. brevis feature domain is therefore a superset of the K. brevis target domain."
* **Why the wider area helps.** *K. brevis* blooms start offshore, 18–74 km from the coast, and winds and currents carry them to shore [2][3][4]. Counts in the wider area can therefore rise before Sarasota's, which makes them useful early-warning features.
* **Matches the satellite data.** The project's chlorophyll file covers lat 26.5–28.0, lon −83.5 to −81.8. The regional box uses the same longitudes and lat 26.3–28.0, so the satellite and cell-count data describe about the same area.

The exact edges of both boxes are a project choice.

**Sources**
1. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729, §2.1–2.2. https://doi.org/10.1016/j.hal.2024.102729
2. FWC Fish and Wildlife Research Institute, *About Florida red tide*: "Florida's red tides begin in nutrient-poor water 18 to 74 kilometers (11 to 46 miles) offshore." https://myfwc.com/research/redtide/general/about/
3. Steidinger, K. A. (2009). Historical perspective on *Karenia brevis* red tide research in the Gulf of Mexico. *Harmful Algae*, 8, 549–561. https://doi.org/10.1016/j.hal.2008.11.009
4. Weisberg, R., Liu, Y., Lembke, C., Hu, C., Hubbard, K., & Garrett, M. (2019). The coastal ocean circulation influence on the 2018 West Florida Shelf *K. brevis* red tide bloom. *Journal of Geophysical Research: Oceans*, 124, 2501–2512. https://doi.org/10.1029/2018JC014887"""))
C.append(code("""h5["in_target"] = is_in_box(h5["LATITUDE"], h5["LONGITUDE"], TARGET_BOX)
print(f"in target box: {h5['in_target'].sum():,} of {len(h5):,} samples")
log_step(name, "of which inside target box", int(h5["in_target"].sum()))

habsos = h5[["date", "LATITUDE", "LONGITUDE", "DESCRIPTION", "CELLCOUNT", "SALINITY", "WATER_TEMP", "in_target"]].rename(columns=str.lower)
habsos.head()"""))

C.append(md("""#### Deliberately *not* done
* **Did not drop rows missing salinity or water temperature** — the cell count (the label) is still valid in those rows.
* **Did not remove high cell counts as outliers** — counts in the millions are real blooms, the thing being predicted.
* **Did not read the label from `CATEGORY`.** It gives the same classes, but `CELLCOUNT` also covers the 856 rows without a category and allows a weekly maximum (1.3).
* **Did not fill gaps in time** — a week with no sample gets no label rather than a guessed one.
* **Did not scale or normalise** — that happens inside each model's training split, so test data never leaks into training."""))

C.append(md("""### 1.7 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name, subset_steps=["of which inside target box"]))
del habsos_raw, sub, fl, h1, h2, h3, h4, dups"""))


nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
out = Path(__file__).with_name("00_master_eda_and_merge.ipynb")
nbf.write(nb, out)
print("wrote", out)
