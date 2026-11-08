"""Unit tests for run_pipeline.py."""
import importlib.util
import sys
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_pipeline_test", MODELS / "run_pipeline.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def test_defaults_come_from_the_settings_file():
    args = rp.parse_args([])
    assert args.horizons == [1, 2, 4] and args.models == ["hybrid_lr", "hybrid_xgb"]


def test_steps_run_cross_validation_then_saved_models_per_horizon():
    steps = rp.list_steps(rp.parse_args(["--horizons", "1", "4", "--models", "hybrid_xgb"]))
    assert [(h, s) for h, s, _ in steps] == [(1, "cross_validation"), (1, "saved_models"),
                                             (4, "cross_validation"), (4, "saved_models")]
    assert steps[0][2][1:] == ["hybrid_escalation/train.py", "hybrid_xgb"]


def test_rebuild_table_comes_first_and_skips_are_honoured():
    steps = rp.list_steps(rp.parse_args(["--horizons", "2", "--rebuild-table", "--skip-cv"]))
    assert [s for _, s, _ in steps] == ["weekly_table", "saved_models"]


def test_unknown_model_is_rejected():
    with pytest.raises(SystemExit):
        rp.parse_args(["--models", "random_forest"])


def test_run_steps_sets_the_horizon_and_times_each_step():
    cmd = [sys.executable, "-c", "import os, sys; sys.exit(os.environ['HAB_HORIZON'] != '4')"]
    times = rp.run_steps([(4, "check", cmd)])
    assert times[0][:2] == (4, "check") and times[0][2] >= 0
