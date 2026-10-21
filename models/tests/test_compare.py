"""Unit tests for compare.py."""
import importlib.util
from pathlib import Path

import pandas as pd

MODELS = Path(__file__).resolve().parents[1]


def load(relpath):
    spec = importlib.util.spec_from_file_location(Path(relpath).stem + "_test", MODELS / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_metrics_stacks_every_model_and_horizon(tmp_path):
    compare = load("compare.py")
    row = {c: 0.5 for c in compare.COLUMNS if c != "horizon"} | {"model": "m", "feature_set": "kb_only",
                                                                 "scheme": "forward"}
    for h in ("h1w", "h4w"):
        (tmp_path / h / "m").mkdir(parents=True)
        pd.DataFrame([row]).to_csv(tmp_path / h / "m" / "metrics.csv", index=False)
    t = compare.collect_metrics(tmp_path)
    assert t["horizon"].tolist() == ["h1w", "h4w"] and list(t.columns) == compare.COLUMNS
