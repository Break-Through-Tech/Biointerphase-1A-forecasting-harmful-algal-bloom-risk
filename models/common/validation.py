"""Validation folds that split the weekly table by bloom year.

Two schemes, both split by August–July bloom year so one red-tide season never sits
in both training and test data (a random split would leak next week's answer,
because neighbouring weeks are nearly identical):

  loyo     leave one bloom year out. Uses the most data, but trains on years
           after the test year.
  forward  forward-chaining: train on every earlier year, test on the next.
           The "deploy it in year Y" estimate. Test years start in 2016, so the
           first training set already contains several bloom seasons.

Chart of the forward-chaining folds: models/figures/forward_chaining_folds.png
"""
from collections.abc import Iterator

import numpy as np
import pandas as pd

from .dates import assign_bloom_year

FORWARD_FIRST_TEST_YEAR = 2016


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

    Training weeks whose label falls inside the test year are purged: with a horizon
    of h weeks, the last h weeks before the test year are labelled with a class
    observed in the test year, so keeping them would train on the test period.

    Args:
        labelled (pandas.DataFrame): Output of select_labelled_rows. If it has a
            "target_week" column (the week each label was observed), purging is applied.
        scheme (str): "loyo" (leave one bloom year out) or "forward" (forward-chaining).

    Returns:
        Iterator[tuple[int, numpy.ndarray, numpy.ndarray]]: For each fold, the test
        bloom year and boolean train and test masks over the rows of `labelled`.

    Raises:
        ValueError: If `scheme` is not "loyo" or "forward".

    Example:
        >>> idx = pd.to_datetime(["2014-09-07", "2015-09-06", "2016-09-04"])
        >>> lab = select_labelled_rows(pd.DataFrame({"target": [0, 1, 2]}, index=idx))
        >>> [(y, tr.tolist(), te.tolist()) for y, tr, te in make_folds(lab, "loyo")]
        [(2015, [False, True, True], [True, False, False]), (2016, [True, False, True], [False, True, False]), (2017, [True, True, False], [False, False, True])]
    """
    if scheme not in ("loyo", "forward"):
        raise ValueError(f"unknown scheme {scheme!r}: use 'loyo' or 'forward'")
    for year in sorted(labelled["bloom_year"].unique()):
        test = labelled["bloom_year"] == year
        if scheme == "loyo":
            train = ~test
        else:
            if year < FORWARD_FIRST_TEST_YEAR:
                continue
            train = labelled["bloom_year"] < year
        if "target_week" in labelled:
            label_year = assign_bloom_year(pd.DatetimeIndex(labelled["target_week"]))
            train = train & ~pd.Series(label_year == year, index=labelled.index)
        yield int(year), train.to_numpy(), test.to_numpy()
