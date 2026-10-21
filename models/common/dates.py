"""Date helpers shared by the data pipeline and the models."""
import pandas as pd


def assign_bloom_year(timestamp_index: pd.DatetimeIndex) -> pd.Index:
    """Assign each date to an August–July "bloom year", named by the year it ends.

    Red tide is seasonal: blooms usually start in late summer or fall and most last
    three to five months (FWC). Bloom years keep each season in one cross-validation
    fold; calendar years would split it across two and leak information.
    Chart of the season: models/figures/bloom_season_by_month.png

    Args:
        timestamp_index (pandas.DatetimeIndex): Dates to label, e.g. the weekly index.

    Returns:
        pandas.Index: Integer bloom year for each date. August–December dates belong
        to the following year's bloom year; January–July dates to the same year's.

    Example:
        >>> assign_bloom_year(pd.DatetimeIndex(["2015-07-26", "2015-08-02", "2016-01-03"])).tolist()
        [2015, 2016, 2016]
    """
    return timestamp_index.year + (timestamp_index.month >= 8).astype(int)
