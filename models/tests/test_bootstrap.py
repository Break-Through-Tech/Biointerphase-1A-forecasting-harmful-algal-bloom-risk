"""Unit tests for experiments/bootstrap_vs_persistence.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODELS = Path(__file__).resolve().parents[1]


def load(relpath):
    spec = importlib.util.spec_from_file_location(Path(relpath).stem + "_test", MODELS / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_compute_metrics():
    boot = load("experiments/bootstrap_vs_persistence.py")
    m = boot.compute_metrics(pd.DataFrame({"y_true": [0, 1, 2], "y_pred": [0, 1, 2]}))
    assert m == {"macro_f1": 1.0, "warning_recall": 1.0}


def test_bootstrap_difference_is_zero_for_a_copy_of_persistence():
    boot = load("experiments/bootstrap_vs_persistence.py")
    joined = pd.DataFrame({"bloom_year": np.repeat([2016, 2017, 2018], 4), "y_true": [0, 1, 2, 0] * 3,
                           "y_pred": [0, 0, 2, 0] * 3})
    joined["y_pers"] = joined["y_pred"]
    d = boot.bootstrap_difference(joined, np.random.default_rng(0), n_boot=50)
    assert d["d_macro_f1"] == 0 and d["p_better_macro_f1"] == 0 and d["d_macro_f1_ci"] == "[+0.000, +0.000]"


def test_bootstrap_difference_favours_a_better_model():
    boot = load("experiments/bootstrap_vs_persistence.py")
    joined = pd.DataFrame({"bloom_year": np.repeat([2016, 2017, 2018], 4), "y_true": [0, 1, 2, 0] * 3,
                           "y_pred": [0, 1, 2, 0] * 3, "y_pers": [0, 0, 0, 0] * 3})
    d = boot.bootstrap_difference(joined, np.random.default_rng(0), n_boot=50)
    assert d["d_macro_f1"] > 0 and d["p_better_macro_f1"] == pytest.approx(1.0)
