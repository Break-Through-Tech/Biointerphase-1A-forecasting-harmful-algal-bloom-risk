"""Save each cleaned project dataset, and the merged weekly table, as CSVs in notebooks/outputs/cleaned/.

Uses the same loaders (common/loaders.py) that build the weekly modelling table, so these
are exactly the rows that go into the merge. CHIRPS and the cyanotoxin file are reviewed
and excluded (see loaders.review_chirps and loaders.review_cyanotoxins), so they are not exported.
"""
from pathlib import Path

from common import loaders as L
from common.cleaning_log import CleaningLog
from common.weekly import build_weekly

OUT = Path(__file__).resolve().parent / "notebooks" / "outputs" / "cleaned"
LOADERS = {
    "habsos_cellcounts_clean.csv": L.load_habsos,
    "chlorophyll_a_clean.csv": L.load_chlorophyll,
    "nutrients_sarasota_clean.csv": L.load_nutrients,
    "bradenton_air_temp_rain_clean.csv": L.load_bradenton,
    "ksrq_airport_weather_clean.csv": L.load_ksrq,
    "sarasota_grid_uwind_clean.csv": L.load_uwind,
    "mote_water_temp_clean.csv": L.load_mote,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()
    for name, load in LOADERS.items():
        df = load(log)
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df):,} rows, {df.shape[1]} columns, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    L.review_chirps(log)
    L.review_cyanotoxins(log)
    log.make_table().to_csv(OUT / "cleaning_log.csv", index=False)

    # Merged weekly table: the §10 merge of 00_master_eda_and_merge.ipynb (same code path as
    # common/weekly.py), with the bloom class 1, 2 and 4 weeks ahead side by side.
    merged = build_weekly()[0].drop(columns=["target", "target_week"])
    for h in (1, 2, 4):
        merged[f"target_{h}w"] = merged["kb_class"].shift(-h)
    merged.index.name = "week_ending"
    merged.to_csv(OUT / "merged_weekly_sarasota.csv")
    print(f"merged_weekly_sarasota.csv: {len(merged):,} weeks, {merged.shape[1]} columns")
