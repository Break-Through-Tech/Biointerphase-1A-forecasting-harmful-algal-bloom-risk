"""Turn results/run_times.csv into results/run_times.md, a readable table of run times.

run_all.sh times every step; this adds the machine and package versions, because the
times depend on both. Run from the models/ folder: python write_run_times.py
"""
import os
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from common.config import MODELS_ROOT

PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost"]


def format_seconds(seconds: float) -> str:
    """Write a duration as seconds, or minutes and seconds.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: e.g. "45 s" or "3 min 20 s".

    Example:
        >>> format_seconds(45), format_seconds(200)
        ('45 s', '3 min 20 s')
    """
    seconds = int(round(seconds))
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min {seconds % 60} s"


def make_run_time_table(times: pd.DataFrame) -> pd.DataFrame:
    """Pivot the run times to one row per step and one column per horizon, plus a total.

    Args:
        times (pandas.DataFrame): Columns horizon_weeks, step and seconds.

    Returns:
        pandas.DataFrame: Formatted durations; rows in run order, with a final "total" row.

    Example:
        >>> t = pd.DataFrame({"horizon_weeks": [1, 1, 4], "step": ["a", "b", "a"], "seconds": [5, 70, 6]})
        >>> make_run_time_table(t).values.tolist()
        [['5 s', '6 s', '11 s'], ['1 min 10 s', '0 s', '1 min 10 s'], ['1 min 15 s', '6 s', '1 min 21 s']]
    """
    wide = times.pivot_table(index="step", columns="horizon_weeks", values="seconds", aggfunc="sum",
                             fill_value=0, sort=False)
    wide.columns = [f"{h} week{'s' if h > 1 else ''} ahead" for h in wide.columns]
    wide["all horizons"] = wide.sum(axis=1)
    wide.loc["total"] = wide.sum()
    return wide.map(format_seconds)


def format_markdown(table: pd.DataFrame) -> str:
    """Write a table as a Markdown table, index first.

    Args:
        table (pandas.DataFrame): Table of strings.

    Returns:
        str: The Markdown table.

    Example:
        >>> print(format_markdown(pd.DataFrame({"1 week ahead": ["5 s"]}, index=["a"])))
        | step | 1 week ahead |
        | --- | --- |
        | a | 5 s |
    """
    rows = [["step", *table.columns], ["---"] * (len(table.columns) + 1)]
    rows += [[str(i), *r] for i, r in zip(table.index, table.values.tolist())]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def describe_machine() -> list[str]:
    """List the machine and package versions the times were measured on.

    Returns:
        list[str]: Markdown bullet lines.
    """
    lines = [f"- Machine: {platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, "
             f"{os.cpu_count()} logical CPUs",
             f"- Python {platform.python_version()}"]
    lines += [f"- {p} {version(p)}" for p in PACKAGES]
    return lines


if __name__ == "__main__":
    results = MODELS_ROOT / "results"
    table = make_run_time_table(pd.read_csv(results / "run_times.csv"))
    text = ["# Run times", "",
            "How long each step of `run_all.sh` took (wall-clock time). The weekly table step rebuilds",
            "the table from the raw data; each model step cross-validates that model on its feature sets",
            "with both schemes and saves the results.", "",
            format_markdown(table), "", "Measured on:", *describe_machine(), ""]
    Path(results / "run_times.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
