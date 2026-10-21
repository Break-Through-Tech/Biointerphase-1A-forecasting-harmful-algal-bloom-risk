"""Unit tests for write_run_times.py."""
import importlib.util
from pathlib import Path

import pandas as pd

MODELS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("write_run_times_test", MODELS / "write_run_times.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_format_seconds():
    assert [rt.format_seconds(s) for s in (0, 59, 60, 3725)] == ["0 s", "59 s", "1 min 0 s", "62 min 5 s"]


def test_make_run_time_table_keeps_run_order_and_adds_totals():
    times = pd.DataFrame({"horizon_weeks": [1, 1, 2, 2], "step": ["weekly_table", "persistence"] * 2,
                          "seconds": [30, 5, 32, 6]})
    t = rt.make_run_time_table(times)
    assert t.index.tolist() == ["weekly_table", "persistence", "total"]
    assert t.columns.tolist() == ["1 week ahead", "2 weeks ahead", "all horizons"]
    assert t.loc["total", "all horizons"] == "1 min 13 s"


def test_format_markdown_has_a_header_and_one_line_per_row():
    md = rt.format_markdown(pd.DataFrame({"a": ["1 s", "2 s"]}, index=["x", "y"]))
    assert md.splitlines() == ["| step | a |", "| --- | --- |", "| x | 1 s |", "| y | 2 s |"]


def test_describe_machine_lists_packages():
    lines = rt.describe_machine()
    assert lines[0].startswith("- Machine:") and any(line.startswith("- scikit-learn") for line in lines)
