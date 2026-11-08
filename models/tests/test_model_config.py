"""Unit tests for common/model_config.py."""
import pytest

from common.model_config import SECTIONS, load_model_config


def test_load_model_config_has_every_section():
    settings = load_model_config()
    assert all(s in settings for s in SECTIONS)
    assert settings["logistic_regression"]["C"] == 0.1
    assert settings["xgboost"]["n_estimators"] == 300


def test_load_model_config_rejects_a_missing_section(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[run]\nhorizons = [1]\n")
    with pytest.raises(KeyError, match="hybrid"):
        load_model_config(path)


def test_train_script_uses_the_settings(monkeypatch):
    import importlib.util
    from pathlib import Path

    from common import model_config

    spec = importlib.util.spec_from_file_location(
        "hybrid_train_settings", Path(__file__).resolve().parents[1] / "hybrid_escalation" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    settings = load_model_config()
    settings["logistic_regression"]["C"] = 2.0
    settings["xgboost"]["max_depth"] = 5
    settings["hybrid"]["beta"] = 2.0
    monkeypatch.setattr(train, "load_model_config", lambda: settings)
    assert train.make_logistic_regression()[-1].C == 2.0
    assert train.make_xgboost().get_params()["max_depth"] == 5
    assert train.make_hybrid_lr().beta == 2.0 and model_config.load_model_config()["hybrid"]["beta"] == 1.0
