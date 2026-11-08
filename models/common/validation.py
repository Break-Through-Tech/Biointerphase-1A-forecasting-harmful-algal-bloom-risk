"""Validation folds that split the weekly table by bloom year.

Folds split by August–July bloom year, so one red-tide season never sits in both
training and test data (a random split would leak next week's answer, because
neighbouring weeks are nearly identical).
"""
import pandas as pd

from .dates import assign_bloom_year


def select_labelled_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the weeks that have a label and add their bloom year.

    Args:
        table (pandas.DataFrame): Weekly table with a "target" column (NaN = no label),
            e.g. the output of weekly.build_weekly.

    Returns:
        pandas.DataFrame: Labelled weeks only, with "target" as int and a "bloom_year" column.

    Example:
        >>> t = pd.DataFrame({"target": [0, None, 2]},
        ...                  index=pd.to_datetime(["2015-07-26", "2015-08-02", "2015-08-09"]))
        >>> select_labelled_rows(t)[["target", "bloom_year"]].values.tolist()
        [[0, 2015], [2, 2016]]
    """
    lab = table[table["target"].notna()].copy()
    lab["target"] = lab["target"].astype(int)
    lab["bloom_year"] = assign_bloom_year(lab.index)
    return lab
