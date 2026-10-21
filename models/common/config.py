"""Project settings shared by the data pipeline and the models."""
import os
from pathlib import Path

# The models/ folder (this file is models/common/config.py).
MODELS_ROOT = Path(__file__).resolve().parents[1]

# Raw data folder: the repo's data/ by default, or HAB_DATA_DIR if set.
DATA_DIR = Path(os.environ.get("HAB_DATA_DIR", MODELS_ROOT.parent / "data"))

CACHE_DIR = MODELS_ROOT / "data"          # cleaned weekly tables (regenerable, not committed)
HORIZON_WEEKS = int(os.environ.get("HAB_HORIZON", 1))  # predict week t+h from data up to week t
RESULTS_DIR = MODELS_ROOT / "results" / f"h{HORIZON_WEEKS}w"  # metrics, predictions, figures per model
