# Cleaned data: merged weekly table and cleaning log

This folder holds the model-ready outputs of the data pipeline:

| File | What it is |
|---|---|
| `merged_weekly_sarasota.csv` | One row per week (weeks end on Sunday), January 2010 to May 2026: 853 weeks, 88 columns |
| `cleaning_log.csv` | Rows kept after each of the 27 cleaning steps across the 9 datasets, as counts and as a share of the raw file |

Both files are stored with Git LFS. Without `git lfs pull` (step 1) they are 3-line pointer files, not data.

## Columns in the merged table

| Columns | What they are |
|---|---|
| 62 features | Grouped by source in `models/common/feature_sets.py`: season, bloom history, satellite chlorophyll, water quality, water temperature, air and rain, wind |
| 20 `*_8w` water-quality columns | Extra characteristics kept for testing; not in any feature set yet |
| `kb_max`, `kb_class` | This week's highest *K. brevis* count in the target box (cells/L) and its class |
| `habsos_water_temp_c` | Water temperature recorded with the HABSOS samples |
| `target_1w`, `target_2w`, `target_4w` | The class observed 1, 2 and 4 weeks later: what the models predict |

Classes: 0 = no bloom (below 10,000 cells/L), 1 = at risk (10,000 to 99,999), 2 = bloom (100,000 or more).
Weeks with no sample in the target box have no class, so their target is empty.

## How to use the table

### 1. Clone the repo and get the data files

```bash
git clone https://github.com/Break-Through-Tech/Biointerphase-1A-forecasting-harmful-algal-bloom-risk.git
cd Biointerphase-1A-forecasting-harmful-algal-bloom-risk
git lfs install
git lfs pull
```

### 2. Install the packages

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 3. Work from the `models/` folder

The code below imports from `models/common/`, so run it with `models/` as the working folder:

```bash
cd models
python          # or: jupyter notebook, or open a notebook in VS Code saved inside models/
```

In Google Colab, run the commands from steps 1 and 2 in a cell with a `!` in front, then `%cd models`.

### 4. Load the table, pick a feature set and a horizon

```python
import pandas as pd
from common.feature_sets import list_feature_columns
from common.dates import assign_bloom_year

t = pd.read_csv("notebooks/outputs/cleaned/merged_weekly_sarasota.csv",
                index_col="week_ending", parse_dates=True)

features = list_feature_columns("kb_only")   # or "env_only", "long", "full"
target = "target_1w"                         # or "target_2w", "target_4w"

data = t.dropna(subset=[target])             # drop weeks with no label
X, y = data[features], data[target]
years = assign_bloom_year(data.index)        # August-July bloom years

print(X.shape)            # (739, 23)
print(y.value_counts())   # 0: 535 weeks, 2: 144, 1: 60
```

| Feature set | Columns | What it tests |
|---|---|---|
| `kb_only` | 23 | Season and bloom history only: does bloom history alone carry the forecast? |
| `env_only` | 31 | Environment only, without current cell counts |
| `long` | 52 | Every source with records from 2010 |
| `full` | 62 | `long` plus wind and Mote data that start in 2019-2022 |

### 5. Split by bloom year

Train on earlier bloom years and test on later ones. Random splits leak, because neighbouring weeks are nearly identical.

```python
train = years <= 2020
X_train, y_train = X[train], y[train]     # 552 weeks, Jan 2010 to Jul 2020
X_test, y_test = X[~train], y[~train]     # 187 weeks, Aug 2020 to Apr 2026
```

### 6. Handle missing values

Some sources start late, so their features are empty in earlier weeks:

| Features | First value |
|---|---|
| Rain and air temperature (Bradenton) | Jan 2015 |
| Gridded wind | Jan 2019 |
| Mote water temperature | Aug 2020 |
| Airport wind and weather (KSRQ) | Jul 2022 |

Use a model that accepts missing values (e.g. XGBoost), or fill them inside the training split only.
Scale features inside each training split too, never on the whole table.

```python
X_full = data[list_feature_columns("full")]
print(X_full.isna().mean().sort_values(ascending=False).head())   # share of weeks missing, by feature
```

### 7. Compare against persistence

Persistence predicts that next week's class equals the last observed class (`last_class`). It is the score a model has to beat.
Most weeks have no bloom, so use macro-F1 and bloom recall, not accuracy alone.

```python
def score(y_true, y_pred):
    """Accuracy, macro-F1 and bloom recall."""
    f1 = []
    for c in (0, 1, 2):
        tp = ((y_pred == c) & (y_true == c)).sum()
        p, r = tp / max((y_pred == c).sum(), 1), tp / max((y_true == c).sum(), 1)
        f1.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return {"accuracy": round(float((y_true == y_pred).mean()), 3),
            "macro_F1": round(float(sum(f1) / 3), 3),
            "bloom_recall": round(float(((y_pred == 2) & (y_true == 2)).sum() / (y_true == 2).sum()), 3)}

print(score(y_test, data.loc[~train, "last_class"]))
# {'accuracy': 0.834, 'macro_F1': 0.65, 'bloom_recall': 0.7}
```

## Rebuild the files

After a change to the raw data or the cleaning steps, run from the `models/` folder:

```bash
python export_cleaned.py
```

This needs the merged satellite chlorophyll file: run `notebooks/merge_chlorophyll.ipynb` (repo root) first if `data/processed/chlorophyll_merged.parquet` does not exist.

## Where the cleaning is explained

Each dataset has a section in `models/notebooks/00_master_eda_and_merge.ipynb` with plots, data-quality checks and the reasoning behind every cleaning step. Section 10 builds this table.
