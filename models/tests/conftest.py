"""Make `common` importable when pytest runs from the repo root or from models/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
