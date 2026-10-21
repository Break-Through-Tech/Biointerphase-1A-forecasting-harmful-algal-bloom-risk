"""Unit tests for common/cleaning_log.py."""
import numpy as np
import pandas as pd

from common.cleaning_log import CleaningLog


def test_record_step_accepts_a_row_count():
    log = CleaningLog()
    log.record_step("ds", "raw", 100)
    assert log.rows == [{"dataset": "ds", "step": "raw", "rows": 100, "note": ""}]


def test_record_step_counts_the_rows_of_a_dataframe():
    log = CleaningLog()
    log.record_step("ds", "raw", pd.DataFrame({"a": [1, 2, 3]}))
    assert log.rows[0]["rows"] == 3


def test_record_step_accepts_numpy_integers_and_keeps_the_note():
    log = CleaningLog()
    log.record_step("ds", "subset", np.int64(7), "inside the box")
    assert log.rows[0]["rows"] == 7 and log.rows[0]["note"] == "inside the box"


def test_make_table_gives_percent_of_each_datasets_first_step():
    log = CleaningLog()
    log.record_step("a", "raw", 200)
    log.record_step("a", "clean", 150)
    log.record_step("b", "raw", 10)
    log.record_step("b", "clean", 9)
    t = log.make_table()
    assert t["pct_of_raw"].tolist() == [100.0, 75.0, 100.0, 90.0]


def test_make_table_of_an_empty_log_has_the_columns_and_no_rows():
    t = CleaningLog().make_table()
    assert t.columns.tolist() == ["dataset", "step", "rows", "note", "pct_of_raw"] and len(t) == 0


def test_make_table_rounds_to_one_decimal():
    log = CleaningLog()
    log.record_step("a", "raw", 3)
    log.record_step("a", "clean", 1)
    assert log.make_table()["pct_of_raw"].tolist() == [100.0, 33.3]
