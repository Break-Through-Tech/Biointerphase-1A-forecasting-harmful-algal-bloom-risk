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

# --- Spatial domain -------------------------------------------------------
# Map of both boxes: models/figures/target_and_regional_boxes.png
# Target box: the area we forecast for, and where the label comes from.
# The Sarasota coast from Anna Maria Island (north) to Englewood (south),
# out to about 45 km offshore. Its busiest sampling sites are the Sarasota Bay
# docks (Bay Dock, New Pass Dock) and Venice (fishing pier and beach).
TARGET_BOX = dict(lat=(26.9, 27.6), lon=(-83.0, -82.3))

# Regional box: a larger area around the target box, used only for "what is
# happening nearby" features (e.g. the highest count in the region last week).
# Blooms start offshore and drift in (FWC), so nearby counts can warn early.
# Medina et al. (2024) use the same design: a feature area that contains the target.
REGIONAL_BOX = dict(lat=(26.3, 28.0), lon=(-83.5, -81.8))

# --- Labels ---------------------------------------------------------------
# The five FWC / HABSOS bloom categories merged into three classes, as in
# Medina, M., Julian, P., Chin, N., & Davis, S. E. (2024). An early-warning forecast
# model for red tide (Karenia brevis) blooms on the southwest coast of Florida.
# Harmful Algae, 139, 102729. https://doi.org/10.1016/j.hal.2024.102729
# Thresholds are cell counts per litre:
#   no bloom : below 10,000      (FWC: not observed, very low)
#   at risk  : 10,000 to 99,999  (FWC: low; respiratory irritation, possible fish kills)
#   bloom    : 100,000 or more   (FWC: medium, high; bloom threshold of Li et al. 2021)
CLASS_EDGES = [0, 1e4, 1e5]
CLASS_NAMES = ["no_bloom", "at_risk", "bloom"]
