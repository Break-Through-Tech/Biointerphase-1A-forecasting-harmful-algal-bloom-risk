"""Read the model settings in models/model_config.toml."""
import tomllib
from pathlib import Path

from .config import MODELS_ROOT

MODEL_CONFIG_PATH = MODELS_ROOT / "model_config.toml"
SECTIONS = ("run", "hybrid", "logistic_regression", "xgboost")


def load_model_config(path: Path | None = None) -> dict:
    """Load the model settings file.

    Args:
        path (pathlib.Path | None): Settings file; models/model_config.toml if None.

    Returns:
        dict: One dict per section: run, hybrid, logistic_regression and xgboost.

    Raises:
        KeyError: If a section is missing.

    Example:
        >>> load_model_config()["run"]["horizons"]
        [1, 2, 4]
    """
    with open(path or MODEL_CONFIG_PATH, "rb") as fh:
        settings = tomllib.load(fh)
    missing = [s for s in SECTIONS if s not in settings]
    if missing:
        raise KeyError(f"model settings file is missing section(s): {', '.join(missing)}")
    return settings
