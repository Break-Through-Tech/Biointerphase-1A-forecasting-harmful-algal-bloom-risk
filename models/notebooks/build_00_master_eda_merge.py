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

# ---------------------------------------------------------------------------
# 2. Chlorophyll
C.append(md("""---
## 2. `ChlorophyllaConcentrationSarasota` + `_Dataset2` — satellite chlorophyll-a

| | File 1 (DS1) `ChlorophyllaConcentrationSarasota` | File 2 (DS2) `…_Dataset2` |
|---|---|---|
| **What** | Gap-filled 2 km satellite chlorophyll-a (mg/m³) | Satellite chlorophyll-a (mg/m³) on a coarser grid |
| **Source** | NOAA CoastWatch multi-sensor DINEOF product [1] | Not documented in the project files |
| **Coverage** | Jun 2020 – Sep 2026 | Jan 2010 – May 2022 |
| **Role** | Bloom biomass signal and chlorophyll anomaly [2][3] | Extends the record back to 2010 |

Chlorophyll measures **all** algae, not only *K. brevis*. NOAA's red-tide detection flags chlorophyll that rises above its recent baseline (a chlorophyll anomaly) [2][3]. The merge method follows `notebooks/merge_chlorophyll.ipynb` in the 1A repo.

**This section:** 2.1 load and first look · 2.2 data-quality checks · 2.3 which pixels are land, and how often is water seen? · 2.4 exploring the data · 2.5 findings · 2.6 cleaning and merging, step by step · 2.7 cleaning summary

**Sources**
1. NOAA CoastWatch, *NOAA MSL12 multi-sensor DINEOF global 2 km gap-filled chlorophyll*. https://coastwatch.noaa.gov/cwn/products/noaa-msl12-multi-sensor-dineof-global-2-km-gap-filled-products-chlorophyll-diffuse.html · Liu, X., & Wang, M. (2023). High spatial resolution gap-free global and regional ocean color products. *IEEE Transactions on Geoscience and Remote Sensing*, 61, 4204118. https://doi.org/10.1109/tgrs.2023.3271465
2. Stumpf, R. P., Culver, M. E., Tester, P. A., et al. (2003). Monitoring *Karenia brevis* blooms in the Gulf of Mexico using satellite ocean color imagery and other data. *Harmful Algae*, 2, 147–160. https://doi.org/10.1016/S1568-9883(02)00083-5
3. Tomlinson, M. C., Wynne, T. T., & Stumpf, R. P. (2009). An evaluation of remote sensing techniques for enhanced detection of the toxic dinoflagellate, *Karenia brevis*. *Remote Sensing of Environment*, 113, 598–609. https://doi.org/10.1016/j.rse.2008.11.003"""))

C.append(md("""### 2.1 Load and first look
One row per **pixel per day**: date, latitude, longitude, chlorophyll-a. Pixels with no reading (cloud, land, sun glint) have an empty value."""))
C.append(code("""chl_raw = {}
for key, stem in [("DS1", "ChlorophyllaConcentrationSarasota"), ("DS2", "ChlorophyllaConcentrationSarasota_Dataset2")]:
    df = read_erddap(stem, usecols=["time", "latitude", "longitude", "chlor_a"],
                     dtype={"latitude": "float32", "longitude": "float32", "chlor_a": "float32"})
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    chl_raw[key] = df.drop(columns="time")
    print(f"{key}: {len(df):,} rows x {df.shape[1]} columns")
chl_raw["DS1"].head(3)"""))
C.append(md("""**Column overview** (DS1, then DS2):"""))
C.append(code("""display(profile_columns(chl_raw["DS1"]))
profile_columns(chl_raw["DS2"])"""))

C.append(md("""### 2.2 Data-quality checks"""))
C.append(code("""def chl_checks(d):
    lats, lons = np.sort(d["latitude"].unique()), np.sort(d["longitude"].unique())
    expected = pd.date_range(d["date"].min(), d["date"].max())
    return {
        "Rows": f"{len(d):,}",
        "Date range": f"{d['date'].min().date()} -> {d['date'].max().date()}",
        "Days missing from the daily series": len(expected) - d["date"].nunique(),
        "Grid (lat x lon)": f"{len(lats)} x {len(lons)}",
        "Grid spacing (degrees)": round(float(np.diff(lats).mean()), 4),
        "Longitude format": f"{lons.min():.2f} to {lons.max():.2f} (0-360 east)",
        "Empty chlorophyll values": f"{d['chlor_a'].isna().mean():.0%}",
        "Duplicate (date, lat, lon)": int(d.duplicated(["date", "latitude", "longitude"]).sum()),
        "Values below 0.01 (outside valid range)": int((d["chlor_a"] < 0.01).sum()),
        "Values at/above 99 (product's upper limit)": int((d["chlor_a"] >= 99).sum()),
        "Min / max chlorophyll": f"{d['chlor_a'].min():.3f} / {d['chlor_a'].max():.1f}",
    }
pd.DataFrame({k: chl_checks(d) for k, d in chl_raw.items()})"""))

C.append(md("""### 2.3 Which pixels are land, and how often is water seen?
Pixels with no value on any day are land cells. The team checked these against a 1 km land map: all of DS1's never-valid cells and 97 % of DS2's are centred on land, and none is entirely water [1][2].

**Sources**
1. Biointerphase 1A, `notebooks/compare_chlorophyll_datasets.ipynb`, "Where do the NaNs come from?".
2. Karin, T. *global-land-mask* (Python package; GLOBE 1 km land/water map). https://github.com/toddkarin/global-land-mask"""))
C.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (key, d) in zip(axes, chl_raw.items()):
    cov = d.groupby(["latitude", "longitude"])["chlor_a"].count().rename("days").reset_index()
    cov["longitude"] = cov["longitude"] - 360
    land = cov[cov["days"] == 0]
    sc = ax.scatter(cov.loc[cov.days > 0, "longitude"], cov.loc[cov.days > 0, "latitude"],
                    c=cov.loc[cov.days > 0, "days"], s=6, cmap="Blues")
    ax.scatter(land["longitude"], land["latitude"], s=6, c="tan", label=f"land ({len(land)} cells)")
    ax.add_patch(plt.Rectangle((TARGET_BOX["lon"][0], TARGET_BOX["lat"][0]), 0.7, 0.7, fill=False, ec="red"))
    ax.set_title(f"{key}: valid days per pixel (red = target box)"); ax.legend(loc="lower left", fontsize=8)
    plt.colorbar(sc, ax=ax)
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Valid days per pixel, one map per file.** The x-axis is longitude and the y-axis is latitude. Each dot is a grid cell, coloured by the number of days it has a value (darker blue = more days). Tan dots never have a value and are treated as land: 558 cells in DS1 and 565 in DS2. The red rectangle is the Sarasota target box.
* Offshore pixels have values on the most days. Pixels next to the coast and inside the bays have far fewer."""))

C.append(md("""### 2.4 Exploring the data
How does chlorophyll vary over time in each file, and do the two agree where they overlap (Jun 2020 – May 2022)?"""))
C.append(code("""OVERLAP_START, OVERLAP_END = pd.Timestamp("2020-06-15"), pd.Timestamp("2022-05-31")
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for key, col in [("DS2", "0.6"), ("DS1", "tab:blue")]:
    d = chl_raw[key]
    m = d.groupby(d["date"].dt.to_period("M"))["chlor_a"].median()
    axes[0].plot(m.index.to_timestamp(), m.values, color=col, label=key, lw=1.3)
axes[0].axvspan(OVERLAP_START, OVERLAP_END, color="orange", alpha=0.15, label="overlap")
axes[0].set_yscale("log"); axes[0].set_title("Monthly median chlorophyll (mg/m³)"); axes[0].legend(frameon=False)
for key, col in [("DS2", "0.6"), ("DS1", "tab:blue")]:
    v = chl_raw[key]["chlor_a"].dropna()
    axes[1].hist(np.log10(v[v > 0]), bins=80, alpha=0.6, density=True, color=col, label=key)
axes[1].set_title("Distribution of log10 chlorophyll"); axes[1].legend(frameon=False)
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: monthly median chlorophyll over time.** The x-axis is time and the y-axis is chlorophyll on a log scale, because values span several orders of magnitude. Grey is DS2, blue is DS1 and the orange band is the overlap. Both files rise and fall each year, and they track each other closely in the overlap.
* **Right: distribution of all values.** The x-axis is log10 chlorophyll (−1 = 0.1 mg/m³, 1 = 10 mg/m³) and the y-axis is the share of values (density). DS1 has more high values than DS2. This is why DS2 is adjusted to DS1's scale before the two are combined (Step 7)."""))
C.append(md("""**Is chlorophyll seasonal?** Pixel values inside the Sarasota target box, grouped by calendar month."""))
C.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
season = {}
for key, col in [("DS2", "0.5"), ("DS1", "tab:blue")]:
    d = chl_raw[key].dropna(subset=["chlor_a"])
    d = d[(d["chlor_a"] >= 0.01) & is_in_box(d["latitude"], d["longitude"] - 360, TARGET_BOX)]
    g = d.groupby(d["date"].dt.month)["chlor_a"]
    season[key] = pd.DataFrame({"median (mg/m³)": g.median(), "% above 5 mg/m³": 100 * g.apply(lambda v: (v > 5).mean())})
    axes[0].plot(season[key].index, season[key]["median (mg/m³)"], "-o", color=col, label=key)
    axes[1].plot(season[key].index, season[key]["% above 5 mg/m³"], "-o", color=col, label=key)
axes[0].set_title("Median chlorophyll by month, target box (mg/m³)"); axes[0].set_xlabel("month")
axes[1].set_title("% of pixel-days above 5 mg/m³, by month"); axes[1].set_xlabel("month")
for ax in axes: ax.set_xticks(range(1, 13)); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
pd.concat(season, axis=1).round(2).T"""))
C.append(md("""**What the figures show**

* **Left: median chlorophyll by month.** The x-axis is the month (1 = January) and the y-axis is the median pixel value in the target box (mg/m³). Both files are lowest in April–June and highest in August–November. In DS1 the median rises from 0.7 mg/m³ in May to 2.4 mg/m³ in September.
* **Right: share of high values by month.** The y-axis is the percentage of pixel-days above 5 mg/m³. High values are rare in April–May (about 5–6 % in DS1, under 1 % in DS2) and most common in September–October (25 % in DS1 in September, 13 % in DS2 in October).
* **The high-chlorophyll months match the red-tide peak** in the HABSOS cell counts (September–November, §1.4). Season (week of year) is therefore included as a feature in §10, so the model can separate the regular seasonal rise from unusual values."""))

C.append(md("""### 2.5 Findings
* **Most raw rows are empty:** 75 % in DS1 and 81 % in DS2.
* **The two files use different grids** (DS1 0.0375°, DS2 0.0417°) **and read differently:** on matching pixel-days in the overlap, DS1 reads 7.5 % lower than DS2 at the median and has more high values. Stacking them unchanged would create a step in 2020.
* **Values sit at the product's limits.** The product's valid range is 0.001–100 mg/m³ [1]. DS1 has 20 values below 0.01 (7 exactly at 0.001), and both files have 10 values at or above 99.
* **Coastal and bay pixels have the fewest observations.**
* **Chlorophyll is seasonal:** lowest in April–June and highest in August–November, the same months as the red-tide peak (§1.4).

**Source**
1. NASA Ocean Biology Processing Group, *chlor_a* product metadata: valid_min = 0.001, valid_max = 100 mg m⁻³ (e.g. NOAA CoastWatch ERDDAP, https://coastwatch.pfeg.noaa.gov/erddap/info/erdMH1chla1day_R2022SQ/index.html)."""))

C.append(md("""### 2.6 Cleaning and merging, step by step
Steps 1–5 clean each file separately, Steps 6–8 merge them, and Step 9 keeps the Sarasota target box."""))
C.append(code("""for key, d in chl_raw.items():
    log_step(f"chlorophyll {key}", "raw file", d)"""))

C.append(md("""#### Step 1 — Convert longitude to −180…180 (rows kept)
Both files store longitude as 0–360° east (e.g. 277.5). Every other dataset uses −180…180 (−82.5), so 360 is subtracted."""))
C.append(code("""for key, d in chl_raw.items():
    d["longitude"] = (d["longitude"] - 360).round(5)
    print(f"{key}: longitude now {d['longitude'].min():.3f} to {d['longitude'].max():.3f}")
    log_step(f"chlorophyll {key}", "longitude converted to -180..180", d, "values changed, rows kept")"""))

C.append(md("""#### Step 2 — Blank values below 0.01 mg/m³ (rows kept)
The chlorophyll algorithm (OCx) is designed for 0.01–100 mg/m³ [1], so values below 0.01 are outside its design range. DS1 has 20 such values, 7 of them exactly at the product minimum of 0.001 (2.5). DS2's lowest value is 0.061, so none of its values are affected. Only the value is set to missing. This follows the team's chlorophyll comparison notebook [2].

**Sources**
1. NASA Ocean Biology Processing Group (2023). *Chlorophyll a — Algorithm Theoretical Basis Document*, v1.1: "The current polynomial expression for OCx spans four orders of magnitude (0.01 to 100 mg m⁻³)." https://oceandata.sci.gsfc.nasa.gov/files/atbd/atbd-obdaac-chlorophyll-a.pdf
2. Biointerphase 1A, `notebooks/compare_chlorophyll_datasets.ipynb` (section 2.5) and `notebooks/merge_chlorophyll.ipynb` (`LOW_LIMIT = 0.01`)."""))
C.append(code("""for key, d in chl_raw.items():
    low = d["chlor_a"] < 0.01
    print(f"{key}: {low.sum()} values below 0.01 -> missing")
    d.loc[low, "chlor_a"] = np.nan
    log_step(f"chlorophyll {key}", "values below 0.01 blanked", d, f"{low.sum()} values set to missing, rows kept")"""))

C.append(md("""#### Step 3 — Mark land cells and flag coastal water cells
* **Land:** cells with no value on any day are land (2.3) and are left out.
* **Distance to land:** for every water cell, the distance to the nearest land cell is computed.
* **Coastal flag:** DS1 water cells within 10 km of land are flagged `coastal`, not removed. Their values are less certain: a coastline pixel mixes land and water, and light from nearby land can contaminate water pixels (the adjacency effect) [1]. They are also observed less often: 19 % of days within 10 km of land, against 38–42 % farther out [2]. The flag keeps these pixels but marks them, so they feed a separate coastal-only chlorophyll feature.

**Sources**
1. Bulgarelli, B., & Zibordi, G. (2018). On the detectability of adjacency effects in ocean color remote sensing of mid-latitude coastal environments by SeaWiFS, MODIS-A, MERIS, OLCI, OLI and MSI. *Remote Sensing of Environment*, 209, 423–438. https://doi.org/10.1016/j.rse.2017.12.021
2. Biointerphase 1A, `notebooks/compare_chlorophyll_datasets.ipynb` (coverage by distance from land) and `notebooks/merge_chlorophyll.ipynb` (`COASTAL_KM = 10`)."""))
C.append(code("""def to_km(lat, lon):
    \"\"\"Convert latitude/longitude to approximate x/y kilometres for distance lookups.

    One degree of latitude is ~111 km; a degree of longitude is shorter by cos(latitude),
    evaluated at Sarasota (~27.3 N). Accurate enough within this small area.

    Args:
        lat (array-like): Latitudes in degrees.
        lon (array-like): Longitudes in degrees.

    Returns:
        numpy.ndarray: Array of shape (n, 2) with [north km, east km] per point.
    \"\"\"
    return np.c_[np.asarray(lat) * 111, np.asarray(lon) * 111 * np.cos(np.radians(27.3))]

water = {}
for key, d in chl_raw.items():
    cells = d.groupby(["latitude", "longitude"])["chlor_a"].count().rename("n_valid").reset_index()
    land, w = cells[cells["n_valid"] == 0], cells[cells["n_valid"] > 0].copy()
    w["km_to_land"] = cKDTree(to_km(land["latitude"], land["longitude"])).query(to_km(w["latitude"], w["longitude"]))[0]
    water[key] = w
    print(f"{key}: {len(w)} water cells, {len(land)} land cells")
    log_step(f"chlorophyll {key}", "land cells identified", d, f"{len(land)} land cells, {len(w)} water cells")
water["DS1"]["coastal"] = (water["DS1"]["km_to_land"] <= 10).astype("int8")
log_step("chlorophyll DS1", "coastal water cells flagged (<= 10 km)", chl_raw["DS1"],
         f"{water['DS1']['coastal'].sum()} of {len(water['DS1'])} water cells flagged, rows kept")
print(f"DS1 coastal water cells (<= 10 km from land): {water['DS1']['coastal'].sum()} of {len(water['DS1'])}")"""))

C.append(md("""#### Step 4 — Keep only valid observations
Once land cells are known, empty pixel-days carry no information and are removed."""))
C.append(code("""chl_obs = {}
for key, d in chl_raw.items():
    chl_obs[key] = d.dropna(subset=["chlor_a"]).reset_index(drop=True)
    show_step(f"chlorophyll {key}", d, chl_obs[key], "valid observations only")
    LOG[-1]["note"] = "empty pixel-days removed (land, cloud, glint)\""""))

C.append(md("""#### Step 5 — Flag values at the upper limit, add a log column (rows kept)
Values of 99 mg/m³ or more are at the product's maximum of 100 (2.5). They are kept but flagged, and left out when the two files are compared in Step 7. `log_chl` = log10(chlorophyll)."""))
C.append(code("""for key, d in chl_obs.items():
    d["chl_at_limit"] = (d["chlor_a"] >= 99).astype("int8")
    d["log_chl"] = np.log10(d["chlor_a"])
    print(f"{key}: {d['chl_at_limit'].sum()} values flagged at limit")
    log_step(f"chlorophyll {key}", "values >= 99 flagged, log10 column added", d, f"{d['chl_at_limit'].sum()} values flagged, rows kept")
ds1, ds2 = chl_obs["DS1"], chl_obs["DS2"]"""))

C.append(md("""#### Step 6 — Put DS2 on DS1's grid
Each DS1 water cell takes the values of its **nearest DS2 water pixel**, if one is within 3.5 km (a project choice, about one DS2 pixel width). DS2 pixels are larger, so one DS2 pixel can feed two DS1 cells, which is why the row count grows. DS1 cells with no DS2 pixel within 3.5 km get no DS2 values."""))
C.append(code("""dist, idx = cKDTree(to_km(water["DS2"]["latitude"], water["DS2"]["longitude"])).query(
    to_km(water["DS1"]["latitude"], water["DS1"]["longitude"]))
grid_map = water["DS1"][["latitude", "longitude"]].assign(
    lat_ds2=water["DS2"]["latitude"].values[idx], lon_ds2=water["DS2"]["longitude"].values[idx], dist_km=dist)
grid_map = grid_map[grid_map["dist_km"] <= 3.5]
ds2_on1 = (ds2.rename(columns={"latitude": "lat_ds2", "longitude": "lon_ds2"})
              .merge(grid_map, on=["lat_ds2", "lon_ds2"]).drop(columns=["lat_ds2", "lon_ds2", "dist_km"]))
print(f"DS1 water cells with a DS2 partner: {len(grid_map)} of {len(water['DS1'])} (median distance {grid_map.dist_km.median():.1f} km)")
show_step("chlorophyll DS2", ds2, ds2_on1, "re-gridded onto DS1 cells")
LOG[-1]["note"] = f"nearest DS2 pixel within 3.5 km; {len(grid_map)} of {len(water['DS1'])} DS1 water cells matched\""""))

C.append(md("""#### Step 7 — Adjust DS2 to read like DS1 (quantile mapping)
Quantile mapping matches two datasets by their percentiles, and is a standard way to remove a systematic difference between two products [1]. On pixel-days where **both** files have a value (the 2020–2022 overlap), DS2's 1st–99th percentiles are matched to DS1's: DS2's 10th percentile becomes DS1's 10th percentile, and so on. Every DS2 value is then translated through that table. Row counts don't change; only DS2's values shift.

**Source**
1. Gudmundsson, L., Bremnes, J. B., Haugen, J. E., & Engen-Skaugen, T. (2012). Technical Note: Downscaling RCM precipitation to the station scale using statistical transformations – a comparison of methods. *Hydrology and Earth System Sciences*, 16, 3383–3390. https://doi.org/10.5194/hess-16-3383-2012"""))
C.append(code("""pairs = ds1.merge(ds2_on1, on=["date", "latitude", "longitude"], suffixes=("_ds1", "_ds2"))
pairs = pairs[(pairs["chl_at_limit_ds1"] == 0) & (pairs["chl_at_limit_ds2"] == 0)]
PCTS = np.linspace(0.01, 0.99, 99)
q2, q1 = np.quantile(pairs["log_chl_ds2"], PCTS), np.quantile(pairs["log_chl_ds1"], PCTS)

def apply_qmap(x: np.ndarray) -> np.ndarray:
    \"\"\"Translate DS2 log-chlorophyll values onto DS1's scale (quantile mapping).

    Values inside the 1st-99th percentile range are interpolated through the
    percentile table; values beyond it are shifted by the end offset instead of
    extrapolated.

    Args:
        x (array-like): DS2 values of log10(chlorophyll-a).

    Returns:
        numpy.ndarray: Adjusted log10(chlorophyll-a) values on DS1's scale.

    Example:
        >>> apply_qmap(np.log10([0.3, 1.0, 5.0])).shape
        (3,)
    \"\"\"
    x = np.asarray(x); y = np.interp(x, q2, q1)
    y = np.where(x < q2[0], x + (q1[0] - q2[0]), y)
    return np.where(x > q2[-1], x + (q1[-1] - q2[-1]), y)

gap = lambda a, b: (1 - 10 ** np.median(a - b)) * 100
print(f"matched pixel-days in the overlap: {len(pairs):,}")
print(f"DS1 reads {gap(pairs['log_chl_ds1'], pairs['log_chl_ds2']):.1f}% lower than DS2 before adjustment, "
      f"{gap(pairs['log_chl_ds1'], apply_qmap(pairs['log_chl_ds2'])):.1f}% after")
display(pd.DataFrame({"percentile": [10, 50, 90, 99],
                      "DS2 value (mg/m³)": 10 ** np.interp([0.1, 0.5, 0.9, 0.99], PCTS, q2),
                      "becomes (mg/m³)": 10 ** np.interp([0.1, 0.5, 0.9, 0.99], PCTS, q1)}).round(3))
ds2_on1["log_chl"] = apply_qmap(ds2_on1["log_chl"])
ds2_on1["chlor_a"] = (10 ** ds2_on1["log_chl"]).astype("float32")
log_step("chlorophyll DS2", "values adjusted to DS1 scale (quantile mapping)", ds2_on1,
         f"DS1-DS2 median gap {gap(pairs['log_chl_ds1'], pairs['log_chl_ds2']):.1f}% -> "
         f"{gap(pairs['log_chl_ds1'], apply_qmap(pairs['log_chl_ds2'])):.1f}%, rows kept")"""))

C.append(md("""#### Step 8 — Combine: DS1 wherever it has a value, adjusted DS2 elsewhere
| Period | Value used |
|---|---|
| 2010 → Jun 2020 | adjusted DS2 |
| Jun 2020 → May 2022 (overlap) | DS1; adjusted DS2 only where DS1 is empty that day |
| Jun 2022 → 2026 | DS1 |"""))
C.append(code("""d2 = ds2_on1.merge(ds1[["date", "latitude", "longitude"]], on=["date", "latitude", "longitude"], how="left", indicator=True)
d2 = d2[d2["_merge"] == "left_only"].drop(columns="_merge")
cols = ["date", "latitude", "longitude", "chlor_a"]
chl_all = (pd.concat([ds1[cols].assign(source="ds1"), d2[cols].assign(source="ds2_adj")], ignore_index=True)
             .merge(water["DS1"][["latitude", "longitude", "coastal"]], on=["latitude", "longitude"], how="left"))
log_step("chlorophyll (merged)", "DS1 + adjusted DS2", chl_all,
         f"{(chl_all['source'] == 'ds1').sum():,} DS1 rows + {(chl_all['source'] == 'ds2_adj').sum():,} adjusted DS2 rows")
print(f"merged rows: {len(chl_all):,} | duplicate (date, lat, lon): {chl_all.duplicated(['date', 'latitude', 'longitude']).sum()}")
pd.crosstab(chl_all["date"].dt.year, chl_all["source"]).T"""))

C.append(md("""#### Step 9 — Keep the Sarasota target box
Only pixels inside the target box (lat 26.9–27.6, lon −83.0 to −82.3; see §1 Step 7) feed the weekly features."""))
C.append(code("""chl = chl_all[is_in_box(chl_all["latitude"], chl_all["longitude"], TARGET_BOX)].copy()
show_step("chlorophyll (merged)", chl_all, chl, "inside target box")
LOG[-1]["note"] = "lat 26.9-27.6, lon -83.0 to -82.3"
m = chl.groupby(chl["date"].dt.to_period("M"))["chlor_a"].median()
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(m.index.to_timestamp(), m.values, color="tab:green")
ax.axvline(OVERLAP_START, color="0.5", lw=1); ax.axvline(OVERLAP_END, color="0.5", lw=1)
ax.set_yscale("log"); ax.set_title("Merged chlorophyll, Sarasota target box: monthly median (grey = overlap edges; no step = merge worked)")
plt.show()"""))
C.append(md("""**What the figure shows**

* **Merged chlorophyll in the target box, monthly median.** The x-axis is time and the y-axis is chlorophyll (mg/m³, log scale). The grey lines mark the start and end of the overlap, where the series switches from adjusted DS2 to DS1.
* The series continues across both lines without a jump, so the adjustment in Step 7 removed the difference between the files."""))

C.append(md("""#### Deliberately *not* done
* **Did not fill cloudy days.** A day with no observation stays missing instead of being interpolated.
* **Did not remove high values.** Bloom-level chlorophyll is the signal.
* **Did not average the two files.** DS1 is the newer, finer product, so it is used wherever it has a value."""))

C.append(md("""### 2.7 Cleaning summary"""))
C.append(code("""for ds in ["chlorophyll DS1", "chlorophyll DS2", "chlorophyll (merged)"]:
    print(ds); display(summarise_cleaning(ds))
del chl_raw, chl_obs, ds1, ds2, ds2_on1, d2, pairs, chl_all"""))

# ---------------------------------------------------------------------------
# 3. Nutrients
C.append(md("""---
## 3. `NutrientsFL_2006_2025` — Sarasota water quality

| | |
|---|---|
| **What** | Lab and field measurements: nitrogen, phosphorus, chlorophyll, temperature, conductivity, oxygen, turbidity, bacteria… |
| **Source** | Sarasota County Water Atlas (USF Water Institute) [1] |
| **Coverage** | 1998 – Dec 2025; programs `SARASOTAES_WQ` (Sarasota County bays) and `SARASOTA_COASTALCREEK_WQ` (tidal creeks) |
| **Role** | Nutrient drivers: nitrogen and phosphorus loading is linked to harmful algal blooms [2] |

**This section:** 3.1 load and first look · 3.2 data-quality checks · 3.3 is each measurement recorded the same way? · 3.4 exploring the data · 3.5 findings · 3.6 cleaning, step by step · 3.7 cleaning summary

**Sources**
1. Sarasota County Water Atlas, *Water quality trends*. https://sarasota.wateratlas.usf.edu/water-quality-trends/
2. Heisler, J., Glibert, P. M., Burkholder, J. M., et al. (2008). Eutrophication and harmful algal blooms: A scientific consensus. *Harmful Algae*, 8, 3–13. https://doi.org/10.1016/j.hal.2008.08.006"""))

C.append(md("""### 3.1 Load and first look
**Long format:** one row per measurement. `Characteristic` says what was measured, and `Result_Value` / `Result_Unit` give the reading."""))
C.append(code("""nut_raw = pd.read_csv(find_file("NutrientsFL_2006_2025"), low_memory=False)
nut_raw["date"] = pd.to_datetime(nut_raw["Activity_Start_Date"], errors="coerce")
print(f"{len(nut_raw):,} rows x {nut_raw.shape[1]} columns")
nut_raw.head(3)"""))
C.append(md("""**Column overview:**"""))
C.append(code("""profile_columns(nut_raw)"""))

C.append(md("""### 3.2 Data-quality checks"""))
C.append(code("""pd.DataFrame([
    ("Programs", str(nut_raw["DataSource"].value_counts().to_dict())),
    ("Stations", nut_raw["Actual_StationID"].nunique()),
    ("Latitude / longitude columns", "none - stations can't be mapped"),
    ("Date range", f"{nut_raw['date'].min().date()} -> {nut_raw['date'].max().date()}"),
    ("Dates that fail to parse", int(nut_raw["date"].isna().sum())),
    ("Distinct characteristics", nut_raw["Characteristic"].nunique()),
    ("Missing Result_Value", int(nut_raw["Result_Value"].isna().sum())),
    ("Activity types", str(nut_raw["Activity_Type"].value_counts().to_dict())),
    ("Depth layers", str(nut_raw["RelativeDepth"].value_counts().to_dict())),
    ("Qualifier codes (U = not detected)", str(nut_raw["Value_Qualifier"].value_counts().head(6).to_dict())),
    ("Exact duplicate rows", int(nut_raw.duplicated().sum())),
], columns=["check", "result"])"""))

C.append(md("""### 3.3 Is each measurement recorded the same way?
Units and value ranges per characteristic. This table shows spelling variants, mixed units and values far outside the normal range."""))
C.append(code("""nut_raw.groupby(["Characteristic", "Result_Unit"])["Result_Value"].agg(["size", "median", "min", "max"]).round(3)"""))

C.append(md("""### 3.4 Exploring the data
How often is the bay sampled, and how do the key variables change over time? The four plotted are the ones the literature emphasises: nitrogen and phosphorus, the nutrients linked to harmful algal blooms [1], and chlorophyll-a, the most consistently significant input for predicting algal blooms across machine-learning methods [2]. Phosphate, one form of phosphorus, is shown alongside the total.

**Sources**
1. Heisler, J., et al. (2008). Eutrophication and harmful algal blooms: A scientific consensus. *Harmful Algae*, 8, 3–13. https://doi.org/10.1016/j.hal.2008.08.006
2. Muttil, N., & Chau, K. W. (2007). Machine-learning paradigms for selecting ecologically significant input variables. *Engineering Applications of Artificial Intelligence*, 20(6), 735–744. https://vuir.vu.edu.au/765/"""))
C.append(code("""plot_per_year(nut_raw["date"].drop_duplicates(), "Distinct sampling times per year (near-continuous sensor data from 2016)")
plt.show()
show = {"Nitrogen": "Total nitrogen (mg/L)", "Phosphorus as P": "Total phosphorus (mg/L)",
        "Phosphorus, phosphate (PO4) as P": "Phosphate (mg/L)", "Chlorophyll a, corrected for pheophytin": "Chlorophyll-a (µg/L)"}
fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
for ax, (ch, title) in zip(axes.ravel(), show.items()):
    s = nut_raw[(nut_raw["Characteristic"] == ch) & (nut_raw["DataSource"] == "SARASOTAES_WQ")]
    m = s.groupby(s["date"].dt.to_period("M"))["Result_Value"].median()
    ax.plot(m.index.to_timestamp(), m.values, color="tab:purple", lw=1)
    ax.set_title(f"Bay stations: {title}, monthly median")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Sampling times per year (bar chart).** The x-axis is the year and the y-axis is the number of distinct sampling timestamps (date and time). Before 2016 there are about 25–80 a year, roughly monthly sampling. From 2016 there are about 900–2,000 a year, from near-continuous sensors.
* **Four panels: monthly medians at the bay stations.** The x-axis is time and the y-axis is the measured value. Total nitrogen and chlorophyll-a rise and fall each year. Total phosphorus drops around 2008 and stays lower afterwards. Phosphate has a gap of about two years around 2008."""))

C.append(md("""### 3.5 Findings
* **21 characteristics in mixed units**, some spelled two ways (`Temperature, water` / `Temperature, Water`; two DO-saturation spellings).
* **Some readings fall far outside the normal range.** 4,965 water-temperature readings are above 100 °F (median 172.5 °F, maximum 412.9 °F), while the Mote sensor in the bay never reads above 33 °C (91 °F, §7). 250 specific-conductance readings are below 100 µS/cm (one is 0), while the typical reading is about 49,000 µS/cm. Step 4 removes them.
* **Sampling changes in 2016**, from about monthly grab samples to near-continuous sensors (3.4). A trailing 8-week median in §10 keeps the two periods comparable.
* **No coordinates.** Stations can only be grouped by program (bay or creek).
* **Values below the detection limit** are marked `U`, and their recorded value is the detection limit itself.
* **7,374 exact duplicate rows** (1.4 %)."""))

C.append(md("""### 3.6 Cleaning, step by step"""))
C.append(code("""name = "NutrientsFL_2006_2025"
log_step(name, "raw file", nut_raw)"""))

C.append(md("""#### Step 1 — Remove exact duplicate rows
7,374 rows (1.4 %) are identical in every column: same station, time, characteristic and value. They are re-entries of the same measurement, and left in they would count twice in weekly medians."""))
C.append(code("""n_dd = nut_raw.drop_duplicates()
show_step(name, nut_raw, n_dd, "drop exact duplicate rows")
LOG[-1]["note"] = "identical in every column (re-entries)\""""))

C.append(md("""#### Step 2 — Standardise characteristic names (all kept)
Each characteristic gets one standard, readable name (e.g. `Phosphorus as P` becomes "total phosphorus"), and spelling variants (e.g. `Temperature, water` / `Temperature, Water`) are merged by lower-casing. All 18 characteristics are kept at this stage.

The most important ones, based on the literature, are **nitrogen and phosphorus** (total N, nitrate + nitrite, total P, phosphate), the nutrients linked to harmful algal blooms [1], and **chlorophyll-a**, the most consistently significant input for predicting algal blooms [2]. Water temperature, conductivity, turbidity and dissolved oxygen describe the conditions around them. The other characteristics (BOD, pH, coliform, TSS, colour, Kjeldahl nitrogen, DO saturation, uncorrected chlorophyll) stay available. Which ones actually help will be tested during modelling, and some will probably be dropped then.

**Sources**
1. Heisler, J., et al. (2008). Eutrophication and harmful algal blooms: A scientific consensus. *Harmful Algae*, 8, 3–13. https://doi.org/10.1016/j.hal.2008.08.006
2. Muttil, N., & Chau, K. W. (2007). Machine-learning paradigms for selecting ecologically significant input variables. *Engineering Applications of Artificial Intelligence*, 20(6), 735–744. https://vuir.vu.edu.au/765/"""))
C.append(code("""NUTRIENT_MAP = {   # file spelling (lower-cased) -> one readable name per characteristic
    "nitrogen": "total nitrogen", "phosphorus as p": "total phosphorus",
    "nitrogen, nitrite (no2) + nitrate (no3) as n": "nitrate + nitrite", "phosphorus, phosphate (po4) as p": "phosphate",
    "nitrogen, ammonia as n": "ammonia", "chlorophyll a, corrected for pheophytin": "chlorophyll-a (corrected)",
    "temperature, water": "water temperature (°F)", "specific conductance": "specific conductance",
    "turbidity": "turbidity", "dissolved oxygen (do)": "dissolved oxygen",
    "bod, biochemical oxygen demand": "biochemical oxygen demand", "chlorophyll a, uncorrected for pheophytin": "chlorophyll-a (uncorrected)",
    "dissolved oxygen saturation": "dissolved oxygen saturation", "fecal coliform": "fecal coliform",
    "nitrogen, kjeldahl": "Kjeldahl nitrogen", "ph": "pH", "total suspended solids (tss)": "total suspended solids",
    "true color": "true colour"}
LITERATURE_PRIORITY = ["total nitrogen", "nitrate + nitrite", "total phosphorus", "phosphate", "chlorophyll-a (corrected)"]
n0 = n_dd.assign(var=n_dd["Characteristic"].str.lower().map(NUTRIENT_MAP))
print("Characteristics without a standard name (should be none):", sorted(n0.loc[n0["var"].isna(), "Characteristic"].unique()))
n1 = n0.dropna(subset=["date", "var", "Result_Value"])
show_step(name, n_dd, n1, "all characteristics, names standardised")
LOG[-1]["note"] = f"{n1['var'].nunique()} characteristics kept; spelling variants merged"
n1["var"].value_counts().rename("rows").to_frame().assign(literature_priority=lambda d: d.index.isin(LITERATURE_PRIORITY))"""))

C.append(md("""#### Step 3 — Replace "not detected" values with half the detection limit (rows kept)
A `U` qualifier means the lab detected nothing, and the value recorded is the detection limit. Replacing it with half the detection limit (DL/2) is a common substitution for values below detection [1].

**Source**
1. Hornung, R. W., & Reed, L. D. (1990). Estimation of average concentration in the presence of nondetectable values. *Applied Occupational and Environmental Hygiene*, 5(1), 46–51. https://doi.org/10.1080/1047322X.1990.10389587"""))
C.append(code("""n2 = n1.copy()
nd = n2["Value_Qualifier"].astype(str).str.contains("U")
print(f"values replaced: {nd.sum():,}")
n2.loc[nd, "Result_Value"] = n2.loc[nd, "Result_Value"] / 2
log_step(name, "not-detected values -> half the detection limit", n2, f"{nd.sum():,} values changed, rows kept")"""))

C.append(md("""#### Step 4 — Remove values outside the expected range (gross range test)
Values outside a minimum–maximum span are removed, following the gross range test for water data [1]. The spans are project choices:

| Measurement | Kept range |
|---|---|
| Water temperature | 40–100 °F |
| Specific conductance | 100–70,000 µS/cm |
| Dissolved oxygen | 0–25 mg/L |
| Chlorophyll-a | 0–1,000 µg/L |
| Total nitrogen | 0–20 mg/L |
| Total phosphorus | 0–5 mg/L |

**Source**
1. U.S. Integrated Ocean Observing System (2020). *Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data*, Version 2.1. https://doi.org/10.25923/x02m-m555, Test 4 (Gross Range Test), p. 17."""))
C.append(code("""VALID = {"water temperature (°F)": (40, 100), "specific conductance": (100, 70000), "dissolved oxygen": (0, 25),
         "chlorophyll-a (corrected)": (0, 1000), "total nitrogen": (0, 20), "total phosphorus": (0, 5)}
bad = np.zeros(len(n2), bool)
for v, (lo, hi) in VALID.items():
    b = ((n2["var"] == v) & ~n2["Result_Value"].between(lo, hi)).to_numpy()
    print(f"{v}: {b.sum():,} values outside {lo}-{hi}")
    bad |= b
n3 = n2[~bad]
show_step(name, n2, n3, "values outside the expected range removed")
LOG[-1]["note"] = "gross range test; " + ", ".join(f"{v} {int(((n2['var'] == v) & ~n2['Result_Value'].between(lo, hi)).sum()):,}" for v, (lo, hi) in VALID.items())"""))

C.append(md("""#### Step 5 — Drop near-bottom readings
The *K. brevis* label uses surface samples only (§1 Step 4), so readings taken near the bottom are dropped to match."""))
C.append(code("""n4 = n3[n3["RelativeDepth"] != "Near Bottom"]
show_step(name, n3, n4, "drop near-bottom readings")
LOG[-1]["note"] = "surface and mid-depth readings kept, to match the surface-only label\""""))

C.append(md("""#### Step 6 — Label each row bay or creek, keep only what the merge needs
The file has no coordinates, so the monitoring program is the only location information:

* **Bay** (`SARASOTAES_WQ`, 40 stations): Sarasota County's coastal bay systems [1].
* **Creek** (`SARASOTA_COASTALCREEK_WQ`, 122 stations): the county's 17 tidally-influenced creeks that connect to those bays [1].

The two are kept as separate features rather than pooled. The table below shows why."""))
C.append(code("""src = n4["DataSource"].map({"SARASOTAES_WQ": "bay", "SARASOTA_COASTALCREEK_WQ": "creek"})
display(n4.assign(source=src).pivot_table(index="var", columns="source", values="Result_Value", aggfunc="median")
          .loc[["specific conductance", "total nitrogen", "total phosphorus"]].round(2).rename_axis("median value"))
tn = n4[n4["var"] == "total nitrogen"].assign(source=src, week=lambda d: d["date"].dt.to_period(WEEK_RULE))
by_src = tn.groupby(["week", "source"])["Result_Value"].median().unstack()
pooled = tn.groupby("week")["Result_Value"].median()
creek_only, bay_only = by_src["creek"].notna() & by_src["bay"].isna(), by_src["bay"].notna() & by_src["creek"].isna()
print(f"weeks with total nitrogen: creek only {creek_only.sum()}, bay only {bay_only.sum()}, both {(by_src.notna().all(axis=1)).sum()}")
print(f"pooled weekly median TN: {pooled[creek_only[creek_only].index].median():.2f} mg/L in creek-only weeks, "
      f"{pooled[bay_only[bay_only].index].median():.2f} mg/L in bay-only weeks")

fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2.2]})
med = n4.assign(source=src).pivot_table(index="var", columns="source", values="Result_Value", aggfunc="median").loc[["specific conductance", "total nitrogen", "total phosphorus"]]
x = np.arange(3)
axes[0].bar(x - 0.2, med["bay"], 0.4, label="bay", color="tab:blue"); axes[0].bar(x + 0.2, med["creek"], 0.4, label="creek", color="tab:orange")
axes[0].set_xticks(x, ["conductance (µS/cm)", "total nitrogen (mg/L)", "total phosphorus (mg/L)"], rotation=20, ha="right"); axes[0].set_yscale("log")
axes[0].set_title("Median value by program (log scale)"); axes[0].legend(frameon=False)
who = pd.Series("both", index=by_src.index); who[creek_only] = "creek only"; who[bay_only] = "bay only"
for lab_, col in [("creek only", "tab:orange"), ("bay only", "tab:blue"), ("both", "0.5")]:
    w = who[who == lab_].index
    axes[1].scatter(w.to_timestamp(), pooled[w], s=8, color=col, label=f"{lab_} ({len(w)} weeks)")
axes[1].set_yscale("log"); axes[1].set_ylabel("mg/L"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("If pooled: weekly median total nitrogen, coloured by which program sampled that week")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show.** Left: median conductance, total nitrogen and total phosphorus for each program; the y-axis is log scale because conductance is in the tens of thousands. Right: what a single pooled weekly nitrogen value would look like. The x-axis is time and the y-axis is the pooled weekly median (mg/L, log scale); each dot is a week, coloured by which program sampled it. Orange (creek-only) weeks sit around 1 mg/L and blue (bay-only) weeks around 0.4, so the pooled series jumps with the sampling, not the water.

* **The water differs.** Median conductance is about 51,000 µS/cm at the bay stations and 820 at the creek stations, and creek total nitrogen and phosphorus are about three times the bay's.
* **They are mostly sampled in different weeks.** Only 149 weeks have total nitrogen from both programs; 511 have creek only and 188 bay only. A pooled weekly median would read about 1.17 mg/L in creek-only weeks and 0.39 mg/L in bay-only weeks. It would jump with whichever program sampled that week, not with any change in the water.
* **Medina et al. also treat land inputs as their own features:** river discharge at two USGS gages and nitrate + nitrite measured at the S-79 structure on the Caloosahatchee River, separate from their coastal measurements (water temperature at the Venice and Fort Myers buoys, sea level at Fort Myers). Peace River discharge and the Caloosahatchee nitrate + nitrite were among the features that ranked relatively highly [2].

So bay values become features for conditions at the coast, and creek values become features for the nutrients coming from land.

**Sources**
1. Sarasota County Water Atlas, *Data & Analysis*: "Sarasota County's 8 major bay systems"; "Sarasota County has 17 tidally-influenced creeks that connect to its coastal bay systems." https://sarasota.wateratlas.usf.edu/analysis
2. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729, §2.2, Fig. 1 and §3. https://doi.org/10.1016/j.hal.2024.102729"""))
C.append(code("""nut = (n4.assign(source=n4["DataSource"].map({"SARASOTAES_WQ": "bay", "SARASOTA_COASTALCREEK_WQ": "creek"}))
          [["date", "source", "var", "Result_Value"]].rename(columns={"Result_Value": "value"}))
log_step(name, "labelled bay or creek; date, source, variable, value kept", nut,
         f"bay {(nut['source'] == 'bay').sum():,} rows, creek {(nut['source'] == 'creek').sum():,} rows")
nut.groupby(["source", "var"])["value"].describe()[["count", "50%", "min", "max"]].round(3)"""))

C.append(md("""#### Deliberately *not* done
* **Did not average stations here.** The weekly median across stations is taken in §10."""))

C.append(md("""### 3.7 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name))
del nut_raw, n_dd, n0, n1, n2, n3, n4"""))

# ---------------------------------------------------------------------------
# 4. Bradenton
C.append(md("""---
## 4. `bradentonFL_tempData` — daily air temperature and rain

| | |
|---|---|
| **What** | Daily precipitation (inches) and max / min / mean air temperature (°F) |
| **Source** | Florida Climate Center (FSU), station 80945 (Bradenton) [1] |
| **Coverage** | 2015 – 2025 |
| **Role** | Rainfall and runoff raise nutrient loads to Florida estuaries [2] |

**This section:** 4.1 load and first look · 4.2 data-quality checks · 4.3 what are the −99.9 values? · 4.4 exploring the data · 4.5 findings · 4.6 cleaning, step by step · 4.7 cleaning summary

**Sources**
1. Florida Climate Center, *Downloadable data*. https://climatecenter.fsu.edu/climate-data-access-tools/downloadable-data
2. Phlips, E. J., Badylak, S., Nelson, N. G., & Havens, K. E. (2020). Hurricanes, El Niño and harmful algal blooms in two sub-tropical Florida estuaries: Direct and indirect impacts. *Scientific Reports*, 10, 1910. https://doi.org/10.1038/s41598-020-58771-4"""))

C.append(md("""### 4.1 Load and first look
One row per day."""))
C.append(code("""brad_raw = pd.read_csv(find_file("bradentonFL_tempData"), skipinitialspace=True)
brad_raw.columns = [c.strip().lower().replace(" ", "_") for c in brad_raw.columns]
brad_raw["date"] = pd.to_datetime(brad_raw[["year", "month", "day"]])
for c in ["precipitation", "max_temp", "min_temp", "mean_temp"]:
    brad_raw[c] = pd.to_numeric(brad_raw[c], errors="coerce")   # blank cells -> NaN
print(f"{len(brad_raw):,} rows x {brad_raw.shape[1]} columns")
display(brad_raw.head(3))
profile_columns(brad_raw)"""))

C.append(md("""### 4.2 Data-quality checks"""))
C.append(code("""ok = brad_raw[["precipitation", "max_temp", "min_temp"]].where(brad_raw[["precipitation", "max_temp", "min_temp"]] > -99)
pd.DataFrame([
    ("Date range", f"{brad_raw['date'].min().date()} -> {brad_raw['date'].max().date()}"),
    ("Duplicate dates", int(brad_raw["date"].duplicated().sum())),
    ("Missing calendar days", len(pd.date_range(brad_raw.date.min(), brad_raw.date.max())) - brad_raw["date"].nunique()),
    ("Sentinel values (<= -99)", str({c: int((brad_raw[c] <= -99).sum()) for c in ["precipitation", "max_temp", "min_temp", "mean_temp"]})),
    ("MEAN TEMP blank", f"{brad_raw['mean_temp'].isna().mean():.0%}"),
    ("Max temp range (°F, excluding sentinels)", f"{ok['max_temp'].min()} to {ok['max_temp'].max()}"),
    ("Min temp range (°F, excluding sentinels)", f"{ok['min_temp'].min()} to {ok['min_temp'].max()}"),
    ("Days where min > max", int((ok["min_temp"] > ok["max_temp"]).sum())),
    ("Largest daily rain (inches)", ok["precipitation"].max()),
], columns=["check", "result"])"""))

C.append(md("""### 4.3 What are the −99.9 values?
The Florida Climate Center states: "Values of -99.9 for temperature or -99.99 for precipitation represent missing data" [1]. Left in, one of them would pull a weekly average temperature down by about 25 °F: a week of 75 °F days averages 75 °F, but with one day recorded as −99.9 it averages (6 × 75 − 99.9) / 7 ≈ 50 °F.

**Source**
1. Florida Climate Center, *Downloadable data*. https://climatecenter.fsu.edu/climate-data-access-tools/downloadable-data"""))
C.append(code("""brad_raw[(brad_raw[["precipitation", "max_temp", "min_temp", "mean_temp"]] <= -99).any(axis=1)]"""))

C.append(md("""### 4.4 Exploring the data"""))
C.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 3.5))
axes[0].plot(brad_raw["date"], (ok["max_temp"] + ok["min_temp"]) / 2, lw=0.5, color="tab:orange")
axes[0].set_title("Daily mean air temperature (°F)")
rain_m = ok.groupby(brad_raw["date"].dt.month)["precipitation"].sum() / brad_raw["date"].dt.year.nunique()
axes[1].bar(rain_m.index, rain_m.values, color="tab:blue"); axes[1].set_title("Average rain per month (inches)")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: daily mean air temperature.** The x-axis is time and the y-axis is °F. It repeats each year, dipping to about 45–60 °F in winter and reaching about 85 °F in summer.
* **Right: average rain per month.** The x-axis is the month (1 = January) and the y-axis is the average total rain for that month, in inches. Rain is highest in June–September (about 7–12.5 inches a month, with August the wettest) and lowest in November–April."""))

C.append(md("""### 4.5 Findings
* **Complete daily record**, no duplicate dates.
* **Missing-data markers:** 7 max-temperature, 8 min-temperature, 5 mean-temperature and 5 precipitation values are −99.9 or −99.99 (4.3). `MEAN TEMP` is also blank on 731 days.
* **Rain is highest June–September**, just before the September–November red-tide peak (§1.4). Rain features are lagged 1–12 weeks in §10.
* **Covers 2015–2025 only**, so rain features are missing for 2010–2014."""))

C.append(md("""### 4.6 Cleaning, step by step"""))
C.append(code("""name = "bradentonFL_tempData"
log_step(name, "raw file", brad_raw)"""))
C.append(md("""#### Step 1 — Missing-data markers → missing (rows kept)
−99.9 and −99.99 are the provider's missing-data codes (4.3), so they are set to missing."""))
C.append(code("""brad = brad_raw.copy()
for c in ["precipitation", "max_temp", "min_temp", "mean_temp"]:
    s = brad[c] <= -99
    print(f"{c}: {s.sum()} markers -> missing")
    brad.loc[s, c] = np.nan
log_step(name, "sentinels -> missing", brad)"""))
C.append(md("""#### Step 2 — Fill blank mean temperature from max and min (rows kept)
In U.S. weather records, the daily average temperature is defined as the average of the day's maximum and minimum [1]. This file follows that definition: every recorded mean temperature (3,282 days) equals (max + min) / 2 exactly. The 731 blank days are filled with the same formula.

**Source**
1. NOAA NCEI, *Local Climatological Data (LCD) Dataset Documentation*: "Average temperature for the day … This is based on the arithmetic average of the maximum and minimum temperature for the day." https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf"""))
C.append(code("""blank = brad["mean_temp"].isna()
brad["mean_temp"] = brad["mean_temp"].fillna((brad["max_temp"] + brad["min_temp"]) / 2)
print(f"mean temp filled: {(blank & brad['mean_temp'].notna()).sum():,} days | still missing: {brad['mean_temp'].isna().sum()}")
log_step(name, "mean temp filled from (max+min)/2", brad)"""))
C.append(md("""#### Deliberately *not* done
* **Did not remove heavy-rain days as outliers.** Large storms deliver the nutrient pulses the rain features are meant to capture [1].

**Source**
1. Phlips, E. J., et al. (2020). Hurricanes, El Niño and harmful algal blooms in two sub-tropical Florida estuaries: Direct and indirect impacts. *Scientific Reports*, 10, 1910. https://doi.org/10.1038/s41598-020-58771-4"""))
C.append(md("""### 4.7 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name))
del brad_raw, ok"""))

# ---------------------------------------------------------------------------
# 5. KSRQ
C.append(md("""---
## 5. `Sarasota_Wind&Temp` — hourly airport weather (KSRQ)

| | |
|---|---|
| **What** | Wind speed and direction, gusts, air and dew-point temperature, pressure, visibility |
| **Source** | NOAA National Weather Service station at Sarasota–Bradenton airport (KSRQ), via SECOORA ERDDAP [1] |
| **Coverage** | Jul 2022 – Jun 2026 |
| **Role** | Wind: *K. brevis* bloom probability is much higher under northerly winds [2] |

**This section:** 5.1 load and first look · 5.2 data-quality checks · 5.3 exploring the data · 5.4 findings · 5.5 cleaning, step by step · 5.6 cleaning summary

**Sources**
1. SECOORA ERDDAP, *gov_noaa_awc_ksrq*. https://erddap.secoora.org/erddap/tabledap/gov_noaa_awc_ksrq.html
2. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))

C.append(md("""### 5.1 Load and first look
One row per hourly weather report."""))
C.append(code("""ksrq_raw = read_erddap("Sarasota_Wind&Temp")
ksrq_raw["date"] = pd.to_datetime(ksrq_raw["time"]).dt.tz_localize(None)
print(f"{len(ksrq_raw):,} rows x {ksrq_raw.shape[1]} columns")
display(ksrq_raw.head(3))
profile_columns(ksrq_raw)"""))

C.append(md("""### 5.2 Data-quality checks"""))
C.append(code("""steps = ksrq_raw["date"].diff().dropna()
pd.DataFrame([
    ("Date range", f"{ksrq_raw['date'].min().date()} -> {ksrq_raw['date'].max().date()}"),
    ("Duplicate timestamps", int(ksrq_raw["date"].duplicated().sum())),
    ("Most common time step", str(steps.mode()[0])),
    ("Gaps longer than 3 hours", int((steps > pd.Timedelta(hours=3)).sum())),
    ("Wind gust missing", f"{ksrq_raw['wind_speed_of_gust'].isna().mean():.0%}"),
    ("Pressure missing", f"{ksrq_raw['air_pressure_at_mean_sea_level'].isna().mean():.0%}"),
    ("QC flag values (all columns)", str(sorted(set(np.ravel(ksrq_raw.filter(like='_qc_agg').values))))),
    ("Wind speed range (m/s)", f"{ksrq_raw['wind_speed'].min()} to {ksrq_raw['wind_speed'].max():.1f}"),
    ("Wind direction outside 0-360", int((~ksrq_raw["wind_from_direction"].dropna().between(0, 360)).sum())),
    ("Air temp range (°C)", f"{ksrq_raw['air_temperature'].min()} to {ksrq_raw['air_temperature'].max()}"),
    ("Air temp outside -5 to 45 °C", int((ksrq_raw["air_temperature"].notna() & ~ksrq_raw["air_temperature"].between(-5, 45)).sum())),
    ("Dew point range (°C)", f"{ksrq_raw['dew_point_temperature'].min()} to {ksrq_raw['dew_point_temperature'].max()}"),
], columns=["check", "result"])"""))

C.append(md("""### 5.3 Exploring the data"""))
C.append(code("""rad = np.deg2rad(ksrq_raw["wind_from_direction"])   # components for this plot only; added as columns in Step 1
comp = pd.DataFrame({"wind_u": -ksrq_raw["wind_speed"] * np.sin(rad), "wind_v": -ksrq_raw["wind_speed"] * np.cos(rad)})
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
axes[0].hist(ksrq_raw["wind_from_direction"].dropna(), bins=36, color="tab:gray"); axes[0].set_title("Wind FROM direction (degrees)")
w = comp.groupby(ksrq_raw["date"].dt.month).mean()
axes[1].plot(w.index, w["wind_u"], "-o", label="u (+ = toward east)"); axes[1].plot(w.index, w["wind_v"], "-o", label="v (+ = toward north)")
axes[1].axhline(0, color="0.5", lw=0.8); axes[1].legend(frameon=False, fontsize=8); axes[1].set_title("Mean wind components by month (m/s)")
axes[2].plot(ksrq_raw["date"], ksrq_raw["air_temperature"], lw=0.3, color="tab:orange"); axes[2].set_title("Air temperature (°C)")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: wind direction.** The x-axis is the direction the wind blows from (0/360 = north, 90 = east, 180 = south, 270 = west) and the y-axis is the number of reports. The tallest bar is at 0°: 3,336 reports have direction 0, and 3,145 of them have a wind speed of 0. Apart from that, easterly winds (about 50–130°) are most common.
* **Middle: mean wind components by month.** The x-axis is the month and the y-axis is m/s. Blue is u (positive = blowing toward the east) and orange is v (positive = blowing toward the north); see Step 1 for how they are computed. Both are negative from September to January, meaning wind from the north-east, and most negative in October.
* **Right: hourly air temperature.** The x-axis is time and the y-axis is °C. The sharp spikes, down to −60 °C and up to 50 °C, are the out-of-range values removed in Step 2."""))
C.append(code("""m = comp.set_index(ksrq_raw["date"]).resample("MS").mean()
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(m.index, m["wind_u"], "-o", ms=3, label="u (+ = toward east)"); ax.plot(m.index, m["wind_v"], "-o", ms=3, label="v (+ = toward north)")
ax.axhline(0, color="0.5", lw=0.8); ax.legend(frameon=False, fontsize=8)
ax.set_title("Airport (KSRQ): monthly mean wind components over time (m/s)"); plt.show()"""))
C.append(md("""**What the figure shows**

* **Monthly mean wind components, July 2022 – June 2026.** The x-axis is time and the y-axis is m/s; blue is u and orange is v.
* Both components are below zero (wind from the north-east) every October–December, in 2022, 2023, 2024 and 2025. The lowest values are in October 2024 (u −2.8, v −1.7 m/s) and October 2025 (u −2.2, v −1.7 m/s)."""))

C.append(md("""### 5.4 Findings
* **Near-complete hourly record**, no duplicate timestamps; 50 gaps longer than 3 hours.
* **Wind gust is 87 % missing** and pressure 18 %.
* **Every QC flag is 2, "not evaluated"** [1], so the flags can't be used to filter.
* **8 air temperatures far outside the normal range:** seven between −60.6 and −6.1 °C on 13–14 May 2024, and one of 50 °C on 30 May 2023.
* **Winds come from the north-east September–January** (u and v both negative, most negative in October). Red tide peaks in the same months (§1.4), so wind and season are hard to separate with about 4 years of data.
* **Starts in July 2022**, and HABSOS sampling in the Sarasota box is near zero in 2022–2023 (§1.5), so there is little overlap with the label.

**Source**
1. U.S. Integrated Ocean Observing System (2020). *Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data*, Version 2.1. https://doi.org/10.25923/x02m-m555, Table 3-1 (flags: 1 = pass, 2 = not evaluated, 3 = suspect, 4 = fail, 9 = missing), p. 14."""))

C.append(md("""### 5.5 Cleaning, step by step"""))
C.append(code("""name = "Sarasota_Wind&Temp (KSRQ)"
log_step(name, "raw file", ksrq_raw)"""))
C.append(md("""#### Step 1 — Split wind direction into north–south and east–west parts (rows kept)
*K. brevis* bloom probability is much higher under northerly winds than southerly winds [1]. Wind direction is an angle, which can't be averaged into weekly values (350° and 10° average to 180°), so each reading is split into a north–south part (`wind_v`) and an east–west part (`wind_u`). Negative `wind_v` = wind from the north. Li et al. use the same split and sign [1].

**Source**
1. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))
C.append(code("""demo = pd.DataFrame({"wind_from_direction": [350, 10], "wind_speed": [5, 5]})
r = np.deg2rad(demo["wind_from_direction"])
u, v = -demo["wind_speed"] * np.sin(r), -demo["wind_speed"] * np.cos(r)
back = (np.rad2deg(np.arctan2(-u.mean(), -v.mean())) + 360) % 360
print(f"plain average of the angles: {demo['wind_from_direction'].mean():.0f} deg (wrong)")
print(f"average via u/v components: {back:.0f} deg (right - still a north wind)")"""))
C.append(code("""rad = np.deg2rad(ksrq_raw["wind_from_direction"])
ksrq_raw["wind_u"] = -ksrq_raw["wind_speed"] * np.sin(rad)   # + = blowing toward the east
ksrq_raw["wind_v"] = -ksrq_raw["wind_speed"] * np.cos(rad)   # + = blowing toward the north; northerly wind -> v < 0
log_step(name, "wind_u and wind_v added", ksrq_raw, "values added, rows kept")
ksrq_raw[["date", "wind_from_direction", "wind_speed", "wind_u", "wind_v"]].head()"""))
C.append(md("""#### Step 2 — Blank out-of-range air and dew-point temperatures (rows kept)
A gross range test [1] with project-chosen spans: air temperature −5 to 45 °C and dew point −10 to 35 °C. It catches the 8 air temperatures listed in 5.4. Only the temperature is set to missing; the wind values in the same reports stay.

**Source**
1. U.S. Integrated Ocean Observing System (2020). *Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data*, Version 2.1. https://doi.org/10.25923/x02m-m555, Test 4 (Gross Range Test), p. 17."""))
C.append(code("""bad_air = ksrq_raw["air_temperature"].notna() & ~ksrq_raw["air_temperature"].between(-5, 45)
bad_dew = ksrq_raw["dew_point_temperature"].notna() & ~ksrq_raw["dew_point_temperature"].between(-10, 35)
display(ksrq_raw.loc[bad_air | bad_dew, ["date", "air_temperature", "dew_point_temperature", "wind_speed"]])
ksrq_raw.loc[bad_air, "air_temperature"] = np.nan
ksrq_raw.loc[bad_dew, "dew_point_temperature"] = np.nan
print(f"air temperatures blanked: {bad_air.sum()} | dew points blanked: {bad_dew.sum()}")
log_step(name, "out-of-range air and dew-point temperatures blanked", ksrq_raw, f"{bad_air.sum()} air temperatures and {bad_dew.sum()} dew points set to missing, rows kept")"""))
C.append(md("""#### Step 3 — Drop reports with neither wind nor air temperature"""))
C.append(code("""k1 = ksrq_raw.dropna(subset=["wind_speed", "air_temperature"], how="all")
show_step(name, ksrq_raw, k1, "reports with neither wind nor air temperature dropped")
LOG[-1]["note"] = f"{len(ksrq_raw) - len(k1):,} reports removed\""""))
C.append(md("""#### Step 4 — Keep only the columns used as features
Dropped: wind gust (87 % missing), visibility (not used as a feature) and the QC-flag columns (all "not evaluated")."""))
C.append(code("""ksrq = k1[["date", "wind_speed", "wind_u", "wind_v", "air_temperature", "dew_point_temperature", "air_pressure_at_mean_sea_level"]]
print("kept columns:", list(ksrq.columns))
log_step(name, "feature columns kept", ksrq, "dropped wind gust (87 % missing), visibility and QC-flag columns; rows kept")"""))
C.append(md("""### 5.6 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name))
del ksrq_raw, k1"""))

# ---------------------------------------------------------------------------
# 6. Gridded wind
C.append(md("""---
## 6. `SarasotaWindData` — gridded east–west wind

| | |
|---|---|
| **What** | East–west (zonal, `uwnd`) wind on a 0.25° grid over the ocean |
| **Source** | CCMP v2.1 near-real-time ocean surface wind [1], via NOAA CoastWatch ERDDAP [2] |
| **Coverage** | Jan 2019 – Sep 2026 |
| **Role** | Extends wind back to 2019, but **east–west only**: the north–south component linked to blooms [3] is not in the file |

**This section:** 6.1 load and first look · 6.2 data-quality checks · 6.3 is the grid regular in space and time? · 6.4 exploring the data · 6.5 findings · 6.6 cleaning, step by step · 6.7 cleaning summary

**Sources**
1. Mears, C. A., Scott, J., Wentz, F. J., Ricciardulli, L., Leidner, S. M., Hoffman, R., & Atlas, R. (2019). A near-real-time version of the Cross-Calibrated Multiplatform (CCMP) ocean surface wind velocity data set. *Journal of Geophysical Research: Oceans*, 124(10), 6997–7010. https://doi.org/10.1029/2019JC015367
2. NOAA CoastWatch ERDDAP, *pifscCcmpDailyV21NRT*. https://coastwatch.pfeg.noaa.gov/erddap/griddap/pifscCcmpDailyV21NRT.html
3. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))

C.append(md("""### 6.1 Load and first look
One row per grid point per time step."""))
C.append(code("""uw_raw = read_erddap("SarasotaWindData")
uw_raw["date"] = pd.to_datetime(uw_raw["time"]).dt.tz_localize(None)
print(f"{len(uw_raw):,} rows x {uw_raw.shape[1]} columns")
display(uw_raw.head(3))
profile_columns(uw_raw)"""))

C.append(md("""### 6.2 Data-quality checks"""))
C.append(code("""pd.DataFrame([
    ("Date range", f"{uw_raw['date'].min().date()} -> {uw_raw['date'].max().date()}"),
    ("Wind components in the file", [c for c in uw_raw.columns if "wnd" in c]),
    ("Missing uwnd", f"{uw_raw['uwnd'].isna().mean():.1%}"),
    ("Duplicate (time, lat, lon)", int(uw_raw.duplicated(["date", "latitude", "longitude"]).sum())),
    ("uwnd range (m/s)", f"{uw_raw['uwnd'].min():.1f} to {uw_raw['uwnd'].max():.1f}"),
    ("Longitude format", f"{uw_raw['longitude'].min()} to {uw_raw['longitude'].max()} (0-360 east)"),
], columns=["check", "result"])"""))

C.append(md("""### 6.3 Is the grid regular in space and time?"""))
C.append(code("""lats, lons = np.sort(uw_raw["latitude"].unique()), np.sort(uw_raw["longitude"].unique())
times = np.sort(uw_raw["date"].unique())
steps = pd.Series(np.diff(times)).value_counts()
print(f"grid: {len(lats)} lat x {len(lons)} lon, spacing {np.diff(lats).mean():.2f} x {np.diff(lons).mean():.2f} degrees")
print(f"time steps: {len(times):,}; step sizes found: {steps.to_dict()}")
print(f"rows per time step: {uw_raw.groupby('date').size().unique().tolist()} (expected {len(lats) * len(lons)})")"""))

C.append(md("""### 6.4 Exploring the data"""))
C.append(code("""m = uw_raw.set_index("date")["uwnd"].resample("MS").mean()
fig, ax = plt.subplots(figsize=(13, 3)); ax.plot(m.index, m.values, color="tab:blue"); ax.axhline(0, color="0.5", lw=0.8)
ax.set_title("Monthly mean east-west wind over the whole grid (m/s; negative = blowing toward the west)"); plt.show()"""))
C.append(md("""**What the figure shows**

* **Monthly mean east–west wind over the grid.** The x-axis is time and the y-axis is m/s. Negative means the wind blows toward the west (from the east).
* The wind is mostly from the east, and most strongly in autumn: the lowest monthly values, −3.1 to −3.7 m/s, are in November 2020, October 2024 and October 2025. This matches the airport station (5.3)."""))
C.append(code("""clim = uw_raw.groupby(uw_raw["date"].dt.month)["uwnd"].mean()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(clim.index, clim.values, color=["tab:blue" if v < 0 else "tab:orange" for v in clim.values]); ax.axhline(0, color="0.5", lw=0.8)
ax.set_xticks(range(1, 13)); ax.set_xlabel("month")
ax.set_title("Mean east-west wind by month, all years (m/s; negative = from the east)"); plt.show()"""))
C.append(md("""**What the figure shows**

* **Mean east–west wind for each calendar month, averaged over 2019–2026.** The x-axis is the month (1 = January) and the y-axis is m/s; negative means wind from the east.
* Every month except July averages below zero. The strongest easterly months are October (−2.4 m/s) and November (−2.1 m/s), the same months the airport station shows wind from the north-east (5.3)."""))

C.append(md("""### 6.5 Findings
* **Regular grid:** no missing values, no duplicates, and every time step has all 49 grid points.
* **9 gaps of 1–2 days** where whole time steps are missing (6.3).
* **East–west component only.** The north–south component, which Li et al. link to bloom probability [1], is not in this file.
* **Wind from the east is strongest in autumn**, matching the airport station (§5).

**Source**
1. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))

C.append(md("""### 6.6 Cleaning, step by step"""))
C.append(code("""name = "SarasotaWindData (u grid)"
log_step(name, "raw file", uw_raw)"""))
C.append(md("""#### Step 1 — Convert longitude to −180…180 (rows kept)"""))
C.append(code("""uw_raw["longitude"] = uw_raw["longitude"] - 360
print(f"longitude now {uw_raw['longitude'].min()} to {uw_raw['longitude'].max()}")
log_step(name, "longitude converted to -180..180", uw_raw, "values changed, rows kept")"""))
C.append(md("""#### Step 2 — Keep the regional box, drop empty values"""))
C.append(code("""uw = uw_raw[is_in_box(uw_raw["latitude"], uw_raw["longitude"], REGIONAL_BOX)].dropna(subset=["uwnd"])
show_step(name, uw_raw, uw, "inside regional box, empty values dropped")
LOG[-1]["note"] = f"{uw[['latitude', 'longitude']].drop_duplicates().shape[0]} of {uw_raw[['latitude', 'longitude']].drop_duplicates().shape[0]} grid points kept\""""))
C.append(md("""#### Step 3 — Average the box into one value per time step
Gives one regional east–west wind series."""))
C.append(code("""uwind = uw.groupby("date", as_index=False)["uwnd"].mean()
log_step(name, "one value per time step (box average)", uwind, f"{len(uwind):,} time steps, {uwind['date'].min().date()} to {uwind['date'].max().date()}")
uwind.head()"""))
C.append(md("""### 6.7 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name))
del uw_raw, uw"""))

# ---------------------------------------------------------------------------
# 7. Mote
C.append(md("""---
## 7. `MoteMarine_BottomTemp` — hourly water temperature

| | |
|---|---|
| **What** | Sea-water temperature at 2 m depth from one fixed sensor near Mote Marine Laboratory (27.33 N, 82.56 W) |
| **Source** | Mote Marine Laboratory, via SECOORA ERDDAP [1] |
| **Coverage** | Aug 2020 – Mar 2025, hourly |
| **Role** | Water temperature is a feature in the Medina et al. forecast [2] |

**This section:** 7.1 load and first look · 7.2 data-quality checks · 7.3 is the hourly record continuous? · 7.4 exploring the data · 7.5 findings · 7.6 cleaning, step by step · 7.7 cleaning summary

**Sources**
1. SECOORA ERDDAP, *org_secoora_scan_rng_sw*. https://erddap.secoora.org/erddap/tabledap/org_secoora_scan_rng_sw.html
2. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729. https://doi.org/10.1016/j.hal.2024.102729"""))

C.append(md("""### 7.1 Load and first look"""))
C.append(code("""mote_raw = read_erddap("MoteMarine_BottomTemp")
mote_raw["date"] = pd.to_datetime(mote_raw["time"]).dt.tz_localize(None)
print(f"{len(mote_raw):,} rows x {mote_raw.shape[1]} columns")
display(mote_raw.head(3))
profile_columns(mote_raw)"""))

C.append(md("""### 7.2 Data-quality checks"""))
C.append(code("""pd.DataFrame([
    ("Date range", f"{mote_raw['date'].min().date()} -> {mote_raw['date'].max().date()}"),
    ("Duplicate timestamps", int(mote_raw["date"].duplicated().sum())),
    ("Locations / depths", f"{mote_raw[['latitude', 'longitude']].drop_duplicates().shape[0]} location, depth {mote_raw['z'].unique().tolist()} m"),
    ("QC flags (1 pass, 3 suspect, 4 fail)", str(mote_raw["sea_water_temperature_qc_agg"].value_counts().to_dict())),
    ("Temperature range (°C)", f"{mote_raw['sea_water_temperature'].min()} to {mote_raw['sea_water_temperature'].max()}"),
    ("Hours missing from the hourly series", len(pd.date_range(mote_raw.date.min(), mote_raw.date.max(), freq="h")) - mote_raw["date"].nunique()),
], columns=["check", "result"])"""))

C.append(md("""### 7.3 Is the hourly record continuous?
Checking for duplicates isn't enough; this looks for jumps between consecutive readings."""))
C.append(code("""t = mote_raw["date"].sort_values()
gaps = t.diff()
big = gaps[gaps > pd.Timedelta(hours=1)]
display(pd.DataFrame({"gap starts": t.shift(1)[big.index], "gap ends": t[big.index],
                      "hours missing": (big / pd.Timedelta(hours=1) - 1).astype(int)}).sort_values("hours missing", ascending=False))"""))

C.append(md("""### 7.4 Exploring the data"""))
C.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 3.5))
axes[0].plot(mote_raw["date"], mote_raw["sea_water_temperature"], lw=0.4, color="tab:red"); axes[0].set_title("Water temperature at 2 m (°C)")
clim = mote_raw.groupby(mote_raw["date"].dt.month)["sea_water_temperature"].mean()
axes[1].bar(clim.index, clim.values, color="tab:red"); axes[1].set_title("Mean by month (°C)")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: hourly water temperature.** The x-axis is time and the y-axis is °C. It cycles each year, from about 12–15 °C in winter to about 31–33 °C in summer. The straight line from March 2023 to March 2024 is the plot joining the two ends of the missing year (7.3), not data.
* **Right: mean temperature by month.** The x-axis is the month and the y-axis is °C. The water is warmest June–September (about 30 °C) and coolest in January (about 17 °C)."""))

C.append(md("""### 7.5 Findings
* **One gap of almost a year:** no data from 22 Mar 2023 to Mar 2024 (8,736 missing hours). Otherwise the record is hourly, from one location and one depth, with no duplicates.
* **Range 11.4–33.0 °C**; 100 readings are flagged suspect or fail.
* **Warmest June–September**, just before the September–November red-tide peak (§1.4).
* **Short record**, so it is combined with the bay water-quality and HABSOS temperatures into one longer series in §10."""))

C.append(md("""### 7.6 Cleaning, step by step"""))
C.append(code("""name = "MoteMarine_BottomTemp"
log_step(name, "raw file", mote_raw)"""))
C.append(md("""#### Step 1 — Keep readings that passed quality control (flag 1)
The flags follow the QARTOD scheme: 1 = pass, 3 = suspect, 4 = fail [1]. Only passing readings are kept.

**Source**
1. U.S. Integrated Ocean Observing System (2020). *Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data*, Version 2.1. https://doi.org/10.25923/x02m-m555, Table 3-1, p. 14."""))
C.append(code("""mote = mote_raw[mote_raw["sea_water_temperature_qc_agg"] == 1][["date", "sea_water_temperature"]]
show_step(name, mote_raw, mote, "readings that passed QC (flag 1) kept")
LOG[-1]["note"] = "removed: " + ", ".join(f"flag {k}: {v}" for k, v in mote_raw.loc[mote_raw["sea_water_temperature_qc_agg"] != 1, "sea_water_temperature_qc_agg"].value_counts().sort_index().items())
log_step(name, "date and temperature columns kept", mote, f"{mote['date'].min().date()} to {mote['date'].max().date()}; 22 Mar 2023 - Mar 2024 gap left unfilled")"""))
C.append(md("""#### Deliberately *not* done
* **Did not fill the 2023–2024 gap.** §10 fills those weeks from the other temperature sources instead."""))
C.append(md("""### 7.7 Cleaning summary"""))
C.append(code("""display(summarise_cleaning(name))
del mote_raw"""))

# ---------------------------------------------------------------------------
# 8. CHIRPS
C.append(md("""---
## 8. `chirps20GlobalDailyP05…` — gridded rainfall (reviewed, excluded)

| | |
|---|---|
| **What** | Daily rainfall estimate (mm/day) on a 0.05° grid |
| **Source** | CHIRPS v2.0, Climate Hazards Center, UC Santa Barbara [1], via ERDDAP |
| **Coverage** | The file spans 2016 – 2026 (see 8.3) |

**This section:** 8.1 load and first look · 8.2 data-quality checks · 8.3 which years actually have data? · 8.4 exploring the data · 8.5 findings · 8.6 decision · 8.7 summary

**Source**
1. Funk, C., Peterson, P., Landsfeld, M., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066. https://doi.org/10.1038/sdata.2015.66"""))

C.append(md("""### 8.1 Load and first look"""))
C.append(code("""chirps = read_erddap("chirps20GlobalDailyP05_0878_d358_0bc6")
chirps["date"] = pd.to_datetime(chirps["time"]).dt.tz_localize(None)
print(f"{len(chirps):,} rows x {chirps.shape[1]} columns")
display(chirps.head(3))
profile_columns(chirps)"""))

C.append(md("""### 8.2 Data-quality checks"""))
C.append(code("""valid = chirps.dropna(subset=["precip"])
cells = chirps.groupby(["latitude", "longitude"])["precip"].count()
pd.DataFrame([
    ("Date range of the file", f"{chirps['date'].min().date()} -> {chirps['date'].max().date()}"),
    ("Missing precipitation", f"{chirps['precip'].isna().mean():.0%}"),
    ("Grid cells that ever have a value (land)", f"{(cells > 0).sum()} of {len(cells)}"),
    ("Duplicate (date, lat, lon)", int(chirps.duplicated(["date", "latitude", "longitude"]).sum())),
    ("Negative rainfall", int((valid["precip"] < 0).sum())),
    ("Max daily rainfall (mm)", round(valid["precip"].max(), 1)),
], columns=["check", "result"])"""))

C.append(md("""### 8.3 Which years actually have data?"""))
C.append(code("""days = valid.groupby(valid["date"].dt.year)["date"].nunique().reindex(range(2016, 2027), fill_value=0)
days.plot(kind="bar", figsize=(10, 2.6), color="tab:blue", title="CHIRPS: days with any rainfall value, per year"); plt.show()
days.to_frame("days with data").T"""))
C.append(md("""**What the figure shows**

* **Days with any rainfall value, per year.** The x-axis is the year and the y-axis is the number of days with at least one value. Only 2016 and 2020 are complete, and 2024 has about half a year. Every other year is empty."""))

C.append(md("""### 8.4 Exploring the data"""))
C.append(code("""land = valid.groupby(["latitude", "longitude"])["precip"].mean().reset_index()
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(land["longitude"], land["latitude"], c=land["precip"], s=8, cmap="Blues")
plt.colorbar(sc, label="mean mm/day"); ax.set_title("CHIRPS: mean daily rain on land cells"); plt.show()"""))
C.append(md("""**What the figure shows**

* **Mean daily rain per grid cell.** The x-axis is longitude and the y-axis is latitude. Each dot is a cell with data, coloured by its mean rain (mm/day, darker = wetter). Only land cells appear; the ocean cells have no values. Inland cells are wetter (about 3 mm/day) than cells near the coast (about 2 mm/day)."""))

C.append(md("""### 8.5 Findings
* **Values exist only for 2016, 2020 and half of 2024.** Every other year in this extract is empty.
* **Ocean cells are always empty**, so 75 % of rows are blank.
* **Where data exists, it has no negative values.**"""))

C.append(md("""### 8.6 Decision — **excluded**
Three scattered years can't support weekly features across 2010–2026. The Bradenton rain gauge (§4) covers 2015–2025 daily and is used instead."""))
C.append(code("""log_step("chirps (rainfall grid)", "raw file", chirps)
log_step("chirps (rainfall grid)", "rows with a rainfall value", valid, "only 2016, 2020 and part of 2024 -> excluded")"""))
C.append(md("""### 8.7 Summary"""))
C.append(code("""display(summarise_cleaning("chirps (rainfall grid)"))
del chirps, valid, cells, land"""))

# ---------------------------------------------------------------------------
# 9. Cyanotoxins
C.append(md("""---
## 9. `florida_cyanotoxins_2015-01-01_to_2026-06-17` — freshwater toxins (reviewed, excluded from the red-tide table)

| | |
|---|---|
| **What** | Lab results for cyanobacteria toxins: microcystins, cylindrospermopsin, anatoxin-a, nodularin, saxitoxins |
| **Source** | Project data link [1]; records come from EPA's STORET database (`ProviderName`), submitted mainly by the South Florida Water Management District and Florida DEP (`Org_FormalName`) |
| **Coverage** | 2015 – 2026, statewide |
| **Role** | A **different HAB type**: toxins from freshwater cyanobacteria ("blue-green algae"), not from *K. brevis* [2] |

**This section:** 9.1 load and first look · 9.2 data-quality checks · 9.3 is this the same kind of bloom as red tide? · 9.4 exploring the data · 9.5 findings · 9.6 decision · 9.7 summary

**Sources**
1. Florida cyanotoxin data story map. https://storymaps.arcgis.com/stories/ccc4c98425194605924ae474fb8002ba
2. U.S. EPA (2019). *Recommended Human Health Recreational Ambient Water Quality Criteria or Swimming Advisories for Microcystins and Cylindrospermopsin*. EPA 822-R-19-001. https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-document-2019.pdf"""))

C.append(md("""### 9.1 Load and first look
One row per toxin result per sample. The file has 65 columns, most of them metadata."""))
C.append(code("""tox = pd.read_csv(find_file("florida_cyanotoxins_2015-01-01_to_2026-06-17"), low_memory=False)
print(f"{len(tox):,} rows x {tox.shape[1]} columns")
keep_cols = ["Location_Type", "Location_Latitude", "Location_Longitude", "Activity_StartDate", "Result_Characteristic",
             "Reslt_Measre", "Result_MeasureUnit", "DetectionLimit_MeasureA"]
display(tox[keep_cols].head(3))
profile_columns(tox[keep_cols])"""))

C.append(md("""### 9.2 Data-quality checks"""))
C.append(code("""mc = tox[tox["Result_Characteristic"].str.contains("Microcystin", na=False)]
vals = pd.to_numeric(mc["Reslt_Measre"], errors="coerce")
dl = pd.to_numeric(mc["DetectionLimit_MeasureA"], errors="coerce")
pd.DataFrame([
    ("Columns entirely empty", int(tox.isna().all().sum())),
    ("Units", str(tox["Result_MeasureUnit"].value_counts().head(3).to_dict())),
    ("Missing result value", int(pd.to_numeric(tox["Reslt_Measre"], errors="coerce").isna().sum())),
    ("Exact duplicate rows", int(tox.duplicated().sum())),
    ("Microcystin results", f"{len(mc):,}"),
    ("...below detection limit or zero", f"{((vals < dl) | (vals == 0)).mean():.1%}"),
    ("...median / max (µg/L)", f"{vals.median()} / {vals.max()}"),
    ("...over EPA recreational limit (8 µg/L)", int((vals > 8).sum())),
], columns=["check", "result"])"""))

C.append(md("""### 9.3 Is this the same kind of bloom as red tide?"""))
C.append(code("""display(tox["Location_Type"].value_counts(normalize=True).round(3).head(6).to_frame("share of samples"))
display(tox["Result_Characteristic"].value_counts().head(8).to_frame("results"))"""))
C.append(md("""**Result.** 66 % of samples are from lakes, 13 % from rivers and streams, 16 % from canals and 3.4 % from estuaries. Each toxin (microcystin variants, cylindrospermopsin, anatoxin-a) is reported about 7,100 times, once per sample."""))

C.append(md("""### 9.4 Exploring the data"""))
C.append(code("""near = tox[is_in_box(tox["Location_Latitude"], tox["Location_Longitude"], REGIONAL_BOX)]
print(f"Rows near Sarasota (regional box): {len(near):,} ->", near["Location_Type"].value_counts().head(4).to_dict())
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(tox["Location_Longitude"], tox["Location_Latitude"], s=2, color="tab:green", alpha=0.3)
ax.add_patch(plt.Rectangle((REGIONAL_BOX["lon"][0], REGIONAL_BOX["lat"][0]), 1.7, 1.7, fill=False, ec="orange"))
ax.set_title("Cyanotoxin sample sites (orange = regional box)"); plt.show()"""))
C.append(md("""**What the figure shows**

* **Where toxin samples were taken.** The x-axis is longitude and the y-axis is latitude. Each green dot is a sample site, and the orange rectangle is the Sarasota regional box. Sites cover the whole state, and most are inland."""))

C.append(md("""### 9.5 Findings
* **66 % of samples are from lakes**, and most of the rest from rivers and canals. Inside the Sarasota regional box: 7,100 rows, mostly rivers, lakes and canals (988 from estuaries).
* **97.6 % of microcystin results are below the detection limit** (median 0.25 µg/L), with rare values up to 780 µg/L. 337 of 72,626 exceed EPA's recommended recreational value of 8 µg/L [1].
* **These toxins come from freshwater cyanobacteria** [1], a different organism from *K. brevis*, measured mostly in different waters.

**Source**
1. U.S. EPA (2019). *Recommended Human Health Recreational Ambient Water Quality Criteria or Swimming Advisories for Microcystins and Cylindrospermopsin*. EPA 822-R-19-001: "For microcystins, the recommended recreational value is 8 micrograms (μg)/liter (L)." https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-document-2019.pdf"""))

C.append(md("""### 9.6 Decision — **excluded from the red-tide table**
It can't label or explain *K. brevis* weeks on the Sarasota coast."""))
C.append(code("""log_step("florida_cyanotoxins", "raw file", tox)
log_step("florida_cyanotoxins", "inside regional box", near, "freshwater toxins; different organism -> excluded")"""))
C.append(md("""### 9.7 Summary"""))
C.append(code("""display(summarise_cleaning("florida_cyanotoxins", subset_steps=["inside regional box"]))
del tox, mc, near"""))

# ---------------------------------------------------------------------------
# 10. Merge
C.append(md("""---
## 10. Merge everything into the weekly modelling table

**One row per week** (weeks end on Sunday), January 2010 – May 2026. Each source is summarised by week and joined on the week.

**No peeking:** the features in week *t* use only data up to the end of week *t*. The label (`target`) is the class observed in week *t* + 1. Lags and rolling windows only look backwards. A week whose following week has no HABSOS sample gets no label.

| Group | Built from | Features |
|---|---|---|
| Bloom history | §1 | weekly maximum, mean and 95th percentile (log), share of samples ≥ 10⁴ and ≥ 10⁵, regional maximum, 1- and 2-week lags, 4- and 8-week maximum, last observed class, weeks since last sample and last bloom |
| Satellite chlorophyll | §2 | weekly mean (log), 90th percentile, share above 1 and 5 mg/m³, coastal mean, chlorophyll anomaly: this week minus the mean of the 8 weeks before, skipping the latest 2 [1] |
| Water quality | §3 | 8-week medians: bay total nitrogen, total phosphorus, nitrate + nitrite, phosphate, chlorophyll-a, specific conductance, turbidity, dissolved oxygen; creek total nitrogen, total phosphorus, nitrate + nitrite |
| Other water quality | §3 | 8-week medians of the remaining 20 bay and creek characteristics (e.g. BOD, pH, Kjeldahl nitrogen, fecal coliform, creek phosphate). In the table for testing; not yet in the models' feature sets |
| Water temperature | §1, §3, §7 | one combined series (Mote, then bay stations, then HABSOS), 2-week change, warm weeks (> 28 °C) in the last 8, HABSOS salinity |
| Air and rain | §4 | mean and maximum air temperature, rain totals over 1, 4, 8 and 12 weeks |
| Wind | §5, §6 | u, v, speed and their 4-week means; airport air temperature and pressure; gridded u |
| Season | — | week of year as sine and cosine |

**Source**
1. Stumpf, R. P., Culver, M. E., Tester, P. A., et al. (2003). Monitoring *Karenia brevis* blooms in the Gulf of Mexico using satellite ocean color imagery and other data. *Harmful Algae*, 2, 147–160. https://doi.org/10.1016/S1568-9883(02)00083-5"""))
C.append(md("""### 10.1 Weekly calendar
One row per week, each ending on a Sunday. Some features look back up to 8 weeks, so the calendar starts in 2008 to give the first weeks of 2010 a full history. The table is trimmed to start in 2010 in 10.9."""))
C.append(code("""end = habsos["date"].max()
weeks = pd.date_range("2008-01-01", end, freq=WEEK_RULE)
print(f"{len(weeks)} weeks: {weeks[0].date()} -> {weeks[-1].date()}")"""))

C.append(md("""### 10.2 Season (2 features)
Computed from each week's date. The week of the year is placed on a circle (sine and cosine), so week 52 sits next to week 1.

* `woy_` = **w**eek **o**f **y**ear · `_sin` / `_cos` = its sine and cosine. Season is included because blooms are seasonal (table below), as are rain (§4), chlorophyll (§2.4) and wind (§5)."""))
C.append(code("""woy = weeks.isocalendar().week.astype(float).to_numpy()
season = pd.DataFrame({"woy_sin": np.sin(2 * np.pi * woy / 52.18),     # 52.18 = average weeks per year (365.25 / 7)
                       "woy_cos": np.cos(2 * np.pi * woy / 52.18)}, index=weeks)
season.assign(week=woy.astype(int)).drop_duplicates("week").set_index("week").loc[[1, 13, 26, 39, 45, 52]].round(2)"""))
C.append(md("""**How often blooms occur in each month** (Sarasota target box, all sampled weeks 2010–2026, by the week's highest cell count):"""))
C.append(code("""wk_max = habsos[habsos["in_target"]].set_index("date")["cellcount"].resample(WEEK_RULE).max().loc[STUDY_START:].dropna()
wk_class = pd.cut(wk_max, bins=CLASS_EDGES + [np.inf], labels=False, right=False)
by_month = wk_class.groupby(wk_class.index.month)
monthly = pd.DataFrame({"sampled weeks": by_month.size(),
                        "bloom %": 100 * by_month.apply(lambda c: (c == 2).mean()),
                        "at risk %": 100 * by_month.apply(lambda c: (c == 1).mean())}).round(0).astype(int)
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(monthly.index, monthly["bloom %"], color="tab:red", label="bloom")
ax.bar(monthly.index, monthly["at risk %"], bottom=monthly["bloom %"], color="tab:orange", label="at risk")
ax.set_xticks(range(1, 13)); ax.set_xlabel("month"); ax.set_ylabel("% of sampled weeks"); ax.legend(frameon=False)
ax.set_title("Share of weeks at bloom / at-risk level, by month (2010-2026)"); plt.show()
monthly.T"""))
C.append(md("""**Result.** Blooms are rare in spring and early summer and common from October to January: 1 % of sampled weeks are at bloom level in May, against 32 % in October, 47 % in November, 39 % in December and 35 % in January. At-risk weeks peak in September (18 %), just before the bloom season."""))

C.append(md("""### 10.3 Bloom history and the weekly class (21 features)
Built from the cleaned HABSOS samples (§1). How to read the names:

* `kb_` = *Karenia brevis* (KB), from samples in the Sarasota target box.
* `rkb_` = regional *K. brevis*, from samples in the wider regional box (§1 Step 5).
* `_log` = log10(cells/L + 1) · `_frac_1e4` / `_frac_1e5` = share of samples at ≥ 10⁴ / ≥ 10⁵ cells/L · `_p95` = 95th percentile · `_L1` / `_L2` = the value 1 or 2 weeks earlier · `_4w` / `_8w` = over the last 4 or 8 weeks.

Each feature and why:

| Features | How | Why |
|---|---|---|
| `kb_max_log`, `kb_mean_log`, `kb_p95_log` | weekly max, mean and 95th percentile of cell counts in the target box, log10(count + 1) | counts span about eight orders of magnitude (§1.4); the weekly max also sets the class |
| `kb_frac_1e4`, `kb_frac_1e5`, `kb_n_samples` | share of the week's samples at ≥ 10⁴ and ≥ 10⁵ cells/L; number of samples | how widespread high counts are, not just the single highest |
| `rkb_max_log`, `rkb_frac_1e5`, `rkb_n_samples` | the same over the wider regional box | blooms start offshore and are carried to shore (§1 Step 7) |
| `_L1`, `_L2`, `_4w`, `_8w` versions | values 1 and 2 weeks earlier; maximum over the last 4 and 8 weeks | Medina et al.'s forecast ranked recent *K. brevis* features highest [1] |
| `last_class`, `weeks_since_sample`, `weeks_since_bloom` | most recent observed class; weeks since the last sample and last bloom week | the class changes slowly from week to week (10.10), so the last known state matters |

`kb_class` is the class of the weekly maximum (no bloom < 10⁴ ≤ at risk < 10⁵ ≤ bloom, §1.3). It becomes the label in 10.9 and is not a feature itself.

**Source**
1. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729. https://doi.org/10.1016/j.hal.2024.102729"""))
C.append(code("""def to_class(cells):
    \"\"\"Convert cell counts to the three red-tide classes.

    Args:
        cells (pandas.Series): K. brevis cells per litre (NaN = not sampled).

    Returns:
        pandas.Series: 0 = no bloom (< 10,000), 1 = at risk (10,000-99,999),
            2 = bloom (>= 100,000); NaN stays NaN.
    \"\"\"
    return pd.cut(cells, bins=CLASS_EDGES + [np.inf], labels=False, right=False)

tgt = habsos[habsos["in_target"]].set_index("date")["cellcount"].resample(WEEK_RULE)
reg = habsos.set_index("date")["cellcount"].resample(WEEK_RULE)
kb = pd.DataFrame(index=weeks)
kb["kb_n_samples"] = tgt.size().reindex(weeks).fillna(0)
kb["kb_max"] = tgt.max().reindex(weeks)
kb["kb_max_log"] = np.log10(kb["kb_max"] + 1)
kb["kb_p95_log"] = np.log10(tgt.quantile(0.95).reindex(weeks) + 1)
kb["kb_mean_log"] = np.log10(tgt.mean().reindex(weeks) + 1)
kb["kb_frac_1e4"] = tgt.apply(lambda s: (s >= 1e4).mean() if len(s) else np.nan).reindex(weeks)
kb["kb_frac_1e5"] = tgt.apply(lambda s: (s >= 1e5).mean() if len(s) else np.nan).reindex(weeks)
kb["rkb_max_log"] = np.log10(reg.max().reindex(weeks) + 1)
kb["rkb_frac_1e5"] = reg.apply(lambda s: (s >= 1e5).mean() if len(s) else np.nan).reindex(weeks)
kb["rkb_n_samples"] = reg.size().reindex(weeks).fillna(0)
kb["kb_class"] = to_class(kb["kb_max"])
kb["last_class"] = kb["kb_class"].ffill()
seen = kb["kb_class"].notna()
kb["weeks_since_sample"] = seen.groupby(seen.cumsum()).cumcount().where(seen.cumsum() > 0)
bloom = kb["kb_class"] == 2
kb["weeks_since_bloom"] = bloom.groupby(bloom.cumsum()).cumcount().astype(float).where(bloom.cumsum() > 0)
for lag in (1, 2):
    for c in ["kb_max_log", "kb_frac_1e5", "rkb_max_log"]:
        kb[f"{c}_L{lag}"] = kb[c].shift(lag)
kb["kb_max_log_4w"] = kb["kb_max_log"].rolling(4, min_periods=1).max()
kb["kb_max_log_8w"] = kb["kb_max_log"].rolling(8, min_periods=1).max()
kb["rkb_max_log_4w"] = kb["rkb_max_log"].rolling(4, min_periods=1).max()
kb.loc["2018-08":"2018-09", ["kb_max", "kb_class", "last_class", "kb_max_log_L1", "kb_max_log_4w", "weeks_since_bloom"]].round(2)"""))

C.append(md("""### 10.4 Satellite chlorophyll (8 features)
Built from the merged chlorophyll pixels in the target box (§2). Each day is summarised over the box, then averaged over the week. How to read the names:

* `chl_` = satellite chlorophyll-a (mg/m³) in the Sarasota target box.
* `_mean_log` = log10 of the mean · `_p90` = 90th percentile · `_frac_gt1` / `_frac_gt5` = share of pixels above 1 / 5 mg/m³ · `_coastal` = only pixels within 10 km of land · `_anomaly` = difference from the recent baseline · `_change_1w` = change from the previous week.

| Features | How | Why |
|---|---|---|
| `chl_mean_log`, `chl_p90` | mean (log10) and 90th percentile of the box | overall level and the high end |
| `chl_frac_gt1`, `chl_frac_gt5` | share of pixels above 1 and 5 mg/m³ | how much of the box is green, not only how green |
| `chl_coastal_mean` | mean over pixels within 10 km of land (§2 Step 3) | the coastal strip, where *K. brevis* is counted |
| `chl_anomaly`, `chl_coastal_anomaly` | this week minus the mean of the 8 weeks before, skipping the latest 2 | NOAA's red-tide flag compares the current image with a 60-day mean ending two weeks earlier [1][2] |
| `chl_mean_log_change_1w` | change in `chl_mean_log` from last week | a sudden rise |

**Sources**
1. Stumpf, R. P., Culver, M. E., Tester, P. A., et al. (2003). Monitoring *Karenia brevis* blooms in the Gulf of Mexico using satellite ocean color imagery and other data. *Harmful Algae*, 2, 147–160. https://doi.org/10.1016/S1568-9883(02)00083-5
2. Tomlinson, M. C. (2004). *Satellite detection and monitoring of harmful algal blooms* (PICES-13 presentation): example anomaly for 22 Sep 2001 against the 60-day mean of 10 Jul – 8 Sep 2001. https://meetings.pices.int/Publications/Presentations/PICES_13/PICES_13_HAB/Tomlinson_HAB.pdf"""))
C.append(code("""daily = chl.groupby("date")["chlor_a"]
d = pd.DataFrame({"chl_mean": daily.mean(), "chl_p90": daily.quantile(0.9),
                  "chl_frac_gt1": daily.apply(lambda s: (s > 1).mean()), "chl_frac_gt5": daily.apply(lambda s: (s > 5).mean())})
d = d.join(chl[chl["coastal"] == 1].groupby("date")["chlor_a"].mean().rename("chl_coastal_mean"))
cf = d.resample(WEEK_RULE).mean().reindex(weeks)
cf["chl_mean_log"] = np.log10(cf["chl_mean"])
cf["chl_anomaly"] = cf["chl_mean"] - cf["chl_mean"].shift(2).rolling(8, min_periods=3).mean()
cf["chl_coastal_anomaly"] = cf["chl_coastal_mean"] - cf["chl_coastal_mean"].shift(2).rolling(8, min_periods=3).mean()
cf["chl_mean_log_change_1w"] = cf["chl_mean_log"].diff()
cf = cf.drop(columns="chl_mean")
cf.loc["2010":].describe().T[["count", "mean", "min", "max"]].round(2)"""))

C.append(md("""### 10.5 Water quality (11 features + 20 for testing)
Built from the cleaned bay and creek measurements (§3), kept separate (§3 Step 6). Each characteristic is first reduced to a weekly median per program, then to a **trailing 8-week median**: before 2016 the bay is sampled about monthly (§3.4), so a single week is usually empty, while 8 weeks usually contain a sample.

How to read the names:

* `bay_` = bay stations (`SARASOTAES_WQ`) · `creek_` = creek stations (`SARASOTA_COASTALCREEK_WQ`) · `_8w` = trailing 8-week median.
* `tn` = total nitrogen · `tp` = total phosphorus · `nox` = nitrate + nitrite · `po4` = phosphate · `nh3` = ammonia · `tkn` = Kjeldahl nitrogen · `chl_insitu` = chlorophyll-a measured in the water (corrected) · `chl_uncorr` = chlorophyll-a, uncorrected · `spcond` = specific conductance · `turbidity` = turbidity · `do` = dissolved oxygen · `do_sat` = dissolved oxygen saturation · `bod` = biochemical oxygen demand · `ph` = pH · `tss` = total suspended solids · `color` = true colour · `fecal_coliform` = fecal coliform.

Feature names use short codes (e.g. `bay_tn_8w`) so this table matches the models' table exactly."""))
C.append(code("""wk = (nut.assign(week=nut["date"].dt.to_period(WEEK_RULE).dt.end_time.dt.normalize())
         .pivot_table(index="week", columns=["source", "var"], values="value", aggfunc="median"))
FEATURE_ABBREV = {"total nitrogen": "tn", "total phosphorus": "tp", "nitrate + nitrite": "nox", "phosphate": "po4",
                  "ammonia": "nh3", "chlorophyll-a (corrected)": "chl_insitu", "water temperature (°F)": "water_temp_f",
                  "specific conductance": "spcond", "turbidity": "turbidity", "dissolved oxygen": "do",
                  "biochemical oxygen demand": "bod", "chlorophyll-a (uncorrected)": "chl_uncorr",
                  "dissolved oxygen saturation": "do_sat", "fecal coliform": "fecal_coliform", "Kjeldahl nitrogen": "tkn",
                  "pH": "ph", "total suspended solids": "tss", "true colour": "color"}
wk = wk[[c for c in wk.columns if c[1] in FEATURE_ABBREV]]
wk.columns = [f"{s}_{FEATURE_ABBREV[v]}" for s, v in wk.columns]
wk = wk.reindex(weeks)
keep = ["bay_tn", "bay_tp", "bay_nox", "bay_po4", "bay_chl_insitu", "bay_spcond", "bay_turbidity", "bay_do",
        "creek_tn", "creek_tp", "creek_nox"]
extra = ["bay_bod", "bay_ph", "bay_tkn", "bay_do_sat", "bay_tss", "bay_color",
         "creek_po4", "creek_nh3", "creek_chl_insitu", "creek_chl_uncorr", "creek_spcond",
         "creek_turbidity", "creek_do", "creek_do_sat", "creek_fecal_coliform", "creek_tkn",
         "creek_bod", "creek_ph", "creek_tss", "creek_color"]
med8 = wk.rolling(8, min_periods=1).median()
nf = pd.concat([med8[keep], med8.reindex(columns=extra)], axis=1).add_suffix("_8w")
bay_water_temp_c = (wk["bay_water_temp_f"] - 32) * 5 / 9
pd.DataFrame({"weeks with a weekly value": wk[keep].loc["2010":].notna().sum(),
              "weeks with an 8-week value": nf[[f"{c}_8w" for c in keep]].loc["2010":].notna().sum().values}).T"""))
C.append(md("""**Result.** Since 2010, a single week has a bay value in only about 190 of 853 weeks; the trailing 8-week median gives a value in about 840."""))

C.append(md("""### 10.6 Water temperature (4 features)
Three sources measure water temperature, each with gaps: the Mote sensor (2020–2025, hourly, §7), the bay stations (§3) and the HABSOS samples (§1). They are combined into one series, using Mote first, then the bay stations, then HABSOS, and gaps of up to 2 weeks are interpolated.

* `_c` = degrees Celsius · `mote_` = Mote sensor · `habsos_` = HABSOS samples in the target box · `_change_2w` = change over the last 2 weeks · `warm_weeks_8w` = number of the last 8 weeks above 28 °C.

| Feature | How |
|---|---|
| `water_temp_c` | the combined weekly series |
| `water_temp_change_2w` | change over the last 2 weeks |
| `warm_weeks_8w` | weeks above 28 °C in the last 8 (28 °C is a project choice) |
| `habsos_salinity` | weekly mean salinity of HABSOS samples in the target box |"""))
C.append(code("""wt = pd.DataFrame(index=weeks)
wt["mote_water_temp_c"] = mote.set_index("date")["sea_water_temperature"].resample(WEEK_RULE).mean().reindex(weeks)
ht = habsos[habsos["in_target"]].set_index("date")
wt["habsos_water_temp_c"] = ht["water_temp"].resample(WEEK_RULE).mean().reindex(weeks)
wt["habsos_salinity"] = ht["salinity"].resample(WEEK_RULE).mean().reindex(weeks)
wt["water_temp_c"] = wt["mote_water_temp_c"].fillna(bay_water_temp_c).fillna(wt["habsos_water_temp_c"]).interpolate(limit=2)
wt["water_temp_change_2w"] = wt["water_temp_c"].diff(2)
wt["warm_weeks_8w"] = (wt["water_temp_c"] > 28).rolling(8, min_periods=1).sum()
src = np.select([wt["mote_water_temp_c"].notna(), bay_water_temp_c.notna(), wt["habsos_water_temp_c"].notna()],
                ["Mote", "bay stations", "HABSOS"], "interpolated or missing")
pd.Series(src, index=weeks).loc["2010":].value_counts().rename("weeks").to_frame()"""))
C.append(md("""**Result.** Since 2010, HABSOS samples supply the temperature in 485 weeks, Mote in 191 and the bay stations in 148; 29 weeks are interpolated or still missing."""))

C.append(md("""### 10.7 Air temperature and rain (6 features)
Built from the Bradenton station (§4, 2015–2025).

* `air_temp_f` = air temperature in °F · `_max` = weekly maximum · `rain_in_Nw` = inches of rain over the last N weeks.

| Features | How | Why |
|---|---|---|
| `air_temp_f`, `air_temp_max_f` | weekly mean and maximum air temperature | weather conditions |
| `rain_in_1w`, `_4w`, `_8w`, `_12w` | rain totals over the last 1, 4, 8 and 12 weeks | rain and runoff raise nutrient loads to estuaries [1], and the effect can arrive weeks later, so several windows are offered |

**Source**
1. Phlips, E. J., Badylak, S., Nelson, N. G., & Havens, K. E. (2020). Hurricanes, El Niño and harmful algal blooms in two sub-tropical Florida estuaries: Direct and indirect impacts. *Scientific Reports*, 10, 1910. https://doi.org/10.1038/s41598-020-58771-4"""))
C.append(code("""b = brad.set_index("date").resample(WEEK_RULE).agg({"precipitation": "sum", "mean_temp": "mean", "max_temp": "max"}).reindex(weeks)
wx = pd.DataFrame(index=weeks)
wx["air_temp_f"], wx["air_temp_max_f"] = b["mean_temp"], b["max_temp"]
for w in (1, 4, 8, 12):
    wx[f"rain_in_{w}w"] = b["precipitation"].rolling(w, min_periods=max(1, w // 2)).sum()
wx.loc["2015":"2025"].describe().T[["count", "mean", "min", "max"]].round(2)"""))

C.append(md("""### 10.8 Wind and airport weather (9 features)
Built from the airport station (§5, from July 2022) and the gridded east–west wind (§6, from 2019). How to read the names:

* `wind_u` = east–west part of the airport wind (m/s, + = blowing toward the east) · `wind_v` = north–south part (+ = toward the north, so a northerly wind is negative) · `wind_speed` in m/s.
* `ksrq_` = the airport station (KSRQ, Sarasota–Bradenton airport) · `pressure_mb` = air pressure in millibars.
* `grid_uwnd` = gridded east–west wind (`uwnd` in the CCMP file, m/s) · `_4w` = mean over the last 4 weeks.

| Features | How | Why |
|---|---|---|
| `wind_u`, `wind_v`, `wind_speed` | weekly means of the airport's east–west and north–south parts (§5 Step 1) and speed | bloom probability is much higher under northerly winds [1] |
| `wind_u_4w`, `wind_v_4w` | 4-week means | wind over a longer period, not only one week (the 4 weeks are a project choice) |
| `ksrq_air_temp_c`, `pressure_mb` | weekly mean airport air temperature and pressure | weather conditions |
| `grid_uwnd`, `grid_uwnd_4w` | weekly and 4-week mean of the gridded east–west wind | the only wind before 2022 |

**Source**
1. Li, M. F., Glibert, P. M., & Lyubchich, V. (2021). Machine learning classification algorithms for predicting *Karenia brevis* blooms on the West Florida Shelf. *Journal of Marine Science and Engineering*, 9(9), 999. https://doi.org/10.3390/jmse9090999"""))
C.append(code("""k = ksrq.set_index("date").resample(WEEK_RULE).mean().reindex(weeks)
wx["wind_u"], wx["wind_v"], wx["wind_speed"] = k["wind_u"], k["wind_v"], k["wind_speed"]
wx["wind_v_4w"] = wx["wind_v"].rolling(4, min_periods=2).mean()
wx["wind_u_4w"] = wx["wind_u"].rolling(4, min_periods=2).mean()
wx["ksrq_air_temp_c"], wx["pressure_mb"] = k["air_temperature"], k["air_pressure_at_mean_sea_level"]
u = uwind.set_index("date")["uwnd"].resample(WEEK_RULE).mean().reindex(weeks)
wx["grid_uwnd"], wx["grid_uwnd_4w"] = u, u.rolling(4, min_periods=2).mean()
wx[["wind_u", "wind_v", "grid_uwnd"]].notna().groupby(wx.index.year).sum().loc[2018:].T"""))

C.append(md("""### 10.9 Join everything and add the label
All groups are joined on the week. The label (`target`) is the class observed `HORIZON_WEEKS` (here 1) week later, so features in week *t* never use data from the week being predicted. The table is then cut to start in January 2010.

* `target` = next week's class: 0 = no bloom, 1 = at risk, 2 = bloom · `target_week` = the week the label comes from · `kb_class` = this week's class (the source of `target`, not a feature) · `kb_max` = this week's highest count in cells/L (not a feature)."""))
C.append(code("""table = pd.concat([kb, cf, nf, wt, wx], axis=1)
table["woy_sin"], table["woy_cos"] = season["woy_sin"], season["woy_cos"]
table["target"] = table["kb_class"].shift(-HORIZON_WEEKS)
table["target_week"] = table.index + pd.Timedelta(weeks=HORIZON_WEEKS)
table = table.loc[STUDY_START:]
print(table.shape, table.index.min().date(), "->", table.index.max().date())
table.head(3)"""))
C.append(md("""**Result.** 853 weeks (3 Jan 2010 – 3 May 2026) and 83 feature columns: 62 in the models' feature sets, the 20 water-quality columns kept for testing, and `habsos_water_temp_c` (used only inside the combined temperature series)."""))
C.append(md("""#### Raw columns left out of the table
Columns from the kept datasets that do not become features, and why:

| Dataset | Column(s) | Why left out |
|---|---|---|
| HABSOS | `CATEGORY` | Same information as `CELLCOUNT`, and blank on 856 rows (§1.3) |
| | `WIND_DIR`, `WIND_SPEED` | Not recorded on over 99 % of samples (QA flag 9 = no data) |
| | `STATE_ID`, `DESCRIPTION`, depth, units, QA and comment columns | Used only for cleaning (QA flag, region box, depth filter) |
| Chlorophyll | `altitude` | Always 0 |
| Nutrients | water temperature | Not its own feature; fills gaps in the water-temperature series (10.6) |
| | station, depth, qualifier and other metadata | Used only for cleaning (duplicates, detection limits, near-bottom readings, bay/creek label) |
| Bradenton | `MIN TEMP` | Used to fill blank mean temperatures (§4 Step 2) |
| | `COOPID`, `YEAR`, `MONTH`, `DAY` | Station ID (the same on every row); the date parts are combined into the date |
| Airport (KSRQ) | wind gust | 87 % missing |
| | visibility, dew point | Not used as features |
| | QC-flag columns | All 2, "not evaluated" (§5.4) |
| Mote | QC flag, latitude, longitude, depth | The flag only filters readings; location and depth are the same on every row |

Gridded wind keeps its only variable (`uwnd`). Every nutrient characteristic is in the table (10.5), including the 20 kept for testing."""))
C.append(md("""### 10.10 Checks on the merged table"""))
C.append(code("""lab = table[table["target"].notna()]
checks = {
    "weeks": len(table),
    "weeks with a label": len(lab),
    "duplicate weeks": int(table.index.duplicated().sum()),
    "label: no_bloom / at_risk / bloom": " / ".join(str(int((lab["target"] == c).sum())) for c in range(3)),
    "label matches class one week later": bool((table["target"].dropna() == table["kb_class"].shift(-1).dropna().reindex(table["target"].dropna().index)).all()),
    "feature columns": table.shape[1] - 4,   # minus kb_max, kb_class, target, target_week
}
display(pd.Series(checks, name="value").to_frame())

groups = {"bloom history": ["kb_max_log", "kb_frac_1e5", "rkb_max_log", "kb_max_log_4w"],
          "satellite chl": ["chl_mean_log", "chl_anomaly", "chl_p90"],
          "water quality": ["bay_tn_8w", "bay_tp_8w", "bay_po4_8w", "bay_chl_insitu_8w"],
          "water temp": ["water_temp_c", "habsos_salinity"],
          "air + rain": ["air_temp_f", "rain_in_4w"],
          "airport wind": ["wind_u", "wind_v"], "gridded wind": ["grid_uwnd"], "Mote temp": ["mote_water_temp_c"]}
cov = pd.DataFrame({g: lab[c].notna().mean(axis=1).groupby(lab.index.year).mean() for g, c in groups.items()})
fig, ax = plt.subplots(figsize=(12, 3.6))
im = ax.imshow(cov.T.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(len(cov)), cov.index); ax.set_yticks(range(len(cov.columns)), cov.columns)
plt.colorbar(im, label="share present"); ax.set_title("Which sources cover which years (labelled weeks)"); ax.grid(False)
plt.show()"""))
C.append(md("""**What the checks show**

* **739 of 853 weeks have a label:** 535 no bloom, 60 at risk, 144 bloom. Every label equals the class one week later, and no week appears twice.
* **Coverage figure.** Rows are feature groups and columns are years; colour is the share of labelled weeks that have that group (dark = always present). Bloom history, satellite chlorophyll, water quality and water temperature cover almost every year. Bloom history thins in 2022–2023, when Sarasota sampling nearly stopped (§1.5). Air and rain start in 2015, gridded wind in 2019, airport wind in 2022, and Mote covers 2020–2025 with its 2023 gap."""))
C.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 3.6), gridspec_kw={"width_ratios": [3, 1]})
colors = table["kb_class"].map({0: "tab:green", 1: "tab:orange", 2: "tab:red"}).fillna("0.85")
axes[0].scatter(table.index, table["kb_max_log"], c=colors, s=7)
axes[0].axhline(4, ls="--", c="0.5", lw=0.8); axes[0].axhline(5, ls="--", c="0.5", lw=0.8)
axes[0].set_title("Weekly max K. brevis in the Sarasota box (green / orange / red = no bloom / at risk / bloom)")
axes[0].set_ylabel("log10 cells/L")
trans = pd.crosstab(lab["last_class"].map(dict(enumerate(CLASS_NAMES))), lab["target"].map(dict(enumerate(CLASS_NAMES))),
                    normalize="index").loc[CLASS_NAMES, CLASS_NAMES]
axes[1].imshow(trans.values, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3): axes[1].text(j, i, f"{trans.values[i, j]:.2f}", ha="center", va="center")
axes[1].set_xticks(range(3), CLASS_NAMES, rotation=20); axes[1].set_yticks(range(3), CLASS_NAMES); axes[1].grid(False)
axes[1].set_title("Last class (rows) -> next week (cols)")
plt.tight_layout(); plt.show()"""))
C.append(md("""**What the figures show**

* **Left: weekly maximum *K. brevis* in the Sarasota box.** The x-axis is time and the y-axis is log10 cells/L; colour is the class (green no bloom, orange at risk, red bloom) and the dashed lines mark 10⁴ and 10⁵. Blooms come in multi-month episodes, for example 2016–2019 and late 2024.
* **Right: how the class changes from one week to the next.** Rows are this week's class and columns are next week's; each row sums to 1. A no-bloom week is followed by another no-bloom week 94 % of the time, and a bloom week by another bloom week 84 % of the time. At-risk weeks are the least predictable: 38 % fall back, 35 % stay and 27 % become blooms.
* **This is why persistence ("next week = this week") is the baseline** every model is compared against, as in Medina et al. [1].

**Source**
1. Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida. *Harmful Algae*, 139, 102729. https://doi.org/10.1016/j.hal.2024.102729"""))
C.append(md("""### 10.11 Which features move with next week's class? (quick screen)
Spearman correlation of each feature with next week's class (0 / 1 / 2). This is a quick screen, not feature selection: tree models can also use combinations of features that a one-by-one correlation misses."""))
C.append(code("""feat_cols = [c for c in table.columns if c not in ("kb_max", "kb_class", "target", "target_week")]
rho = lab[feat_cols].corrwith(lab["target"], method="spearman").dropna()
screen = pd.DataFrame({"spearman_rho": rho, "weeks_with_data": lab[rho.index].notna().sum()})
screen.reindex(rho.abs().sort_values(ascending=False).index).head(20).round(3)"""))
C.append(md("""**Result.** The 18 strongest features are all bloom history (ρ 0.61–0.84). The strongest environmental features are the 4-week mean east–west wind (ρ −0.49, but only 135 labelled weeks) and bay phosphate (ρ 0.42, 726 weeks)."""))
C.append(md("""### 10.12 Cleaning log — how much of each dataset survives
Every logged step from §1–§9 in one table. `pct_of_first_step` is the rows remaining as a share of that dataset's first step."""))
C.append(code("""log = pd.DataFrame(LOG)
log["pct_of_first_step"] = log.groupby("dataset")["rows"].transform(lambda s: (100 * s / s.iloc[0]).round(1))
log"""))
C.append(md("""### 10.13 Save"""))
C.append(code("""table.to_csv(OUTPUT_DIR / "weekly_sarasota_h1.csv")
log.to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)
try:
    table.to_parquet(OUTPUT_DIR / "weekly_sarasota_h1.parquet")
except ImportError:
    pass
print("saved to", OUTPUT_DIR, "->", sorted(p.name for p in OUTPUT_DIR.iterdir()))"""))

C.append(md("""---
## Summary of decisions

| Dataset | Decision | Main reason |
|---|---|---|
| HABSOS | **Keep**, rebuilt from the raw file | The only *K. brevis* source: gives the label and bloom history. Rows missing salinity or temperature are kept |
| Satellite chlorophyll | **Keep**, DS1 + quantile-mapped DS2 | Environmental record from 2010; chlorophyll anomaly feature |
| NutrientsFL | **Keep**, duplicates removed, bay and creek separate | Nitrogen and phosphorus drivers; all characteristics kept for later testing |
| Bradenton | **Keep** | Rain (lagged) and air temperature, 2015–2025 |
| KSRQ airport | **Keep**, wind as u/v | Northerly wind raises bloom probability; only from July 2022 |
| Gridded wind | **Keep** | East–west only, from 2019 |
| Mote | **Keep**, part of one water-temperature series | Short record with a one-year gap |
| CHIRPS | **Exclude** | Only 2016, 2020 and half of 2024 |
| Cyanotoxins | **Exclude** from the red-tide table | Freshwater cyanobacteria, a different organism |

**No scaling here.** Features are scaled inside each model's training split, so test-period statistics never reach training. Extreme cell counts are kept because they are the blooms."""))


nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
out = Path(__file__).with_name("00_master_eda_and_merge.ipynb")
nbf.write(nb, out)
print("wrote", out)
