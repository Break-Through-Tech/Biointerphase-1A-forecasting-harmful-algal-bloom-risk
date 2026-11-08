"""Validation folds that split the weekly table by bloom year.

Both schemes split by August–July bloom year, so one red-tide season never sits in
both training and test data (a random split would leak next week's answer, because
neighbouring weeks are nearly identical):

  forward  forward-chaining, the main scheme: for each test year from 2016 onward,
           train on every earlier year and test on that year. This matches real use:
           train on the past, forecast the coming year.
  loyo     leave one bloom year out: each year is tested once, training on all the
           others, including later years. Used to compare with Medina et al. (2024),
           who split their data the same way.

Chart of the forward-chaining folds: models/figures/forward_chaining_folds.png
"""
from collections.abc import Iterator

import numpy as np
import pandas as pd

from .dates import assign_bloom_year

FORWARD_FIRST_TEST_YEAR = 2016
SCHEMES = ("forward", "loyo")


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


def make_folds(labelled: pd.DataFrame, scheme: str) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield the train and test weeks of every fold.

    Training weeks whose label falls inside a test year are purged: with a horizon of
    h weeks, the last h weeks before a test year are labelled with a class observed in
    that year, so keeping them would train on the test period.

    Args:
        labelled (pandas.DataFrame): Output of select_labelled_rows. If it has a
            "target_week" column (the week each label was observed), purging is applied.
        scheme (str): "forward" or "loyo" (see the module docstring).

    Returns:
        Iterator[tuple[int, numpy.ndarray, numpy.ndarray]]: For each fold, its first
        test bloom year and boolean train and test masks over the rows of `labelled`.

    Raises:
        ValueError: If `scheme` is not one of SCHEMES.

    Example:
        >>> idx = pd.to_datetime(["2014-09-07", "2015-09-06", "2016-09-04"])
        >>> lab = select_labelled_rows(pd.DataFrame({"target": [0, 1, 2]}, index=idx))
        >>> [(y, tr.tolist(), te.tolist()) for y, tr, te in make_folds(lab, "forward")]
        [(2016, [True, False, False], [False, True, False]), (2017, [True, True, False], [False, False, True])]
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}: use one of {SCHEMES}")
    years = sorted(labelled["bloom_year"].unique())
    if scheme == "forward":
        splits = [[y] for y in years if y >= FORWARD_FIRST_TEST_YEAR]
    else:
        splits = [[y] for y in years]
    label_year = (assign_bloom_year(pd.DatetimeIndex(labelled["target_week"]))
                  if "target_week" in labelled else None)
    for test_years in splits:
        test = labelled["bloom_year"].isin(test_years)
        train = ~test if scheme == "loyo" else labelled["bloom_year"] < min(test_years)
        if label_year is not None:
            train = train & ~pd.Series(np.isin(label_year, test_years), index=labelled.index)
        yield int(min(test_years)), train.to_numpy(), test.to_numpy()
