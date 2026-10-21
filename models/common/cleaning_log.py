"""The cleaning log: how many rows of each dataset survive every cleaning step."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningLog:
    """Row counts after every cleaning step, for every dataset.

    Attributes:
        rows (list[dict]): One dict per step: dataset, step, rows, note.

    Example:
        >>> log = CleaningLog()
        >>> log.record_step("habsos_cellcounts", "raw", 219152)
        >>> log.make_table()[["dataset", "step", "rows", "pct_of_raw"]].values.tolist()
        [['habsos_cellcounts', 'raw', 219152, 100.0]]
    """
    rows: list = field(default_factory=list)

    def record_step(self, dataset: str, step: str, df_or_n: pd.DataFrame | int, note: str = "") -> None:
        """Record how many rows a dataset has after one cleaning step.

        Args:
            dataset (str): Dataset name, e.g. "habsos_cellcounts".
            step (str): Short description of the step, e.g. "drop duplicates".
            df_or_n (pandas.DataFrame | int): The data after the step, or its row count.
            note (str): Optional extra detail shown in the cleaning log.

        Returns:
            None

        Example:
            >>> log = CleaningLog()
            >>> log.record_step("habsos_cellcounts", "QA flag == 1", 218960, "192 Texas rows removed")
            >>> log.rows[0]["rows"]
            218960
        """
        n = df_or_n if isinstance(df_or_n, (int, np.integer)) else len(df_or_n)
        self.rows.append(dict(dataset=dataset, step=step, rows=int(n), note=note))

    def make_table(self) -> pd.DataFrame:
        """Return the log as a table, with each step's rows as a percentage of the raw file.

        Returns:
            pandas.DataFrame: Columns dataset, step, rows, note and pct_of_raw
            (rows ÷ the dataset's first logged step × 100, rounded to 0.1).

        Example:
            >>> log = CleaningLog()
            >>> log.record_step("mote", "raw", 31729); log.record_step("mote", "QC pass", 31629)
            >>> log.make_table()["pct_of_raw"].tolist()
            [100.0, 99.7]
        """
        t = pd.DataFrame(self.rows, columns=["dataset", "step", "rows", "note"])
        t["pct_of_raw"] = t.groupby("dataset")["rows"].transform(lambda s: (100 * s / s.iloc[0]).round(1))
        return t
