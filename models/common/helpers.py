"""Small helper functions shared by the loaders and the weekly feature builder."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_EDGES


def is_in_box(lat: pd.Series, lon: pd.Series, box: dict) -> pd.Series:
    """Flag points that fall inside a latitude/longitude box (edges included).

    Args:
        lat (pandas.Series): Latitudes in degrees north.
        lon (pandas.Series): Longitudes in degrees east, −180 to 180.
        box (dict): {"lat": (south, north), "lon": (west, east)}, e.g. TARGET_BOX.

    Returns:
        pandas.Series: Boolean, True where the point is inside the box.

    Example:
        >>> box = {"lat": (26.9, 27.6), "lon": (-83.0, -82.3)}
        >>> is_in_box(pd.Series([27.3, 28.5]), pd.Series([-82.6, -82.6]), box).tolist()
        [True, False]
    """
    return lat.between(*box["lat"]) & lon.between(*box["lon"])


def read_erddap(path: Path, **kw) -> pd.DataFrame:
    """Read an ERDDAP CSV export, skipping the units row under the header.

    ERDDAP puts a row of units (e.g. "UTC", "degrees_north") directly under the header;
    read as data it would turn every column into text.

    Args:
        path (pathlib.Path): CSV file exported from an ERDDAP server.
        **kw: Passed on to pandas.read_csv (e.g. usecols).

    Returns:
        pandas.DataFrame: The data without the units row.

    Example:
        >>> from .config import DATA_DIR
        >>> read_erddap(DATA_DIR / "MoteMarine_BottomTemp.csv").columns[:2].tolist()
        ['time', 'latitude']
    """
    return pd.read_csv(path, skiprows=[1], **kw)


def assign_class(cells: pd.Series) -> pd.Series:
    """Convert K. brevis counts to the three bloom classes.

    Args:
        cells (pandas.Series): Cell counts in cells per litre (NaN = not sampled).

    Returns:
        pandas.Series: 0 = no bloom (< 10,000), 1 = at risk (10,000–99,999),
        2 = bloom (≥ 100,000); NaN stays NaN.

    Example:
        >>> assign_class(pd.Series([0, 9999, 10000, 250000, np.nan])).tolist()
        [0.0, 0.0, 1.0, 2.0, nan]
    """
    return pd.cut(cells, bins=CLASS_EDGES + [np.inf], labels=False, right=False)
