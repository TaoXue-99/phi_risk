import os

import pytest

from phl_risk.exceptions import ExperimentError, OptionalDependencyError
from phl_risk.modeling.experiment.adapters.hydra import compose_config
from phl_risk.modeling.experiment.adapters.yaml import read_yaml_config


def test_plain_yaml_and_hydra_are_model_independent(tmp_path):
    pytest.importorskip("yaml")
    pytest.importorskip("hydra")
    path = tmp_path / "tuning.yaml"
    path.write_text("max_depth: 3\nnum_leaves: 7\n")
    assert read_yaml_config(path) == {"max_depth": 3, "num_leaves": 7}
    cwd = os.getcwd()
    composed = compose_config(config_dir=tmp_path, config_name="tuning", overrides=["max_depth=2"])
    assert composed.config["max_depth"] == 2
    assert composed.overrides == ("max_depth=2",)
    assert os.getcwd() == cwd
    path.write_text("seed: 2026\nother: ${seed}\n")
    with pytest.raises(ExperimentError, match="compose_config"):
        read_yaml_config(path)
    assert compose_config(config_dir=tmp_path, config_name="tuning").config["other"] == 2026


def test_missing_hydra(monkeypatch, tmp_path):
    import importlib

    original = importlib.import_module

    def missing(name, *a, **k):
        if name == "hydra":
            raise ImportError("no hydra")
        return original(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(OptionalDependencyError):
        compose_config(config_dir=tmp_path, config_name="missing")


def test_independent_configuration_sets_and_generator_changes(tmp_path):
    pytest.importorskip("hydra")
    path = tmp_path / "baseline.yaml"
    text = "params:\n  max_depth: 4\n  learning_rate: 0.05\n  metric: [auc, binary_logloss]\n"
    path.write_text(text)
    baseline = compose_config(config_dir=tmp_path, config_name="baseline")
    depth3 = compose_config(
        config_dir=tmp_path,
        config_name="baseline",
        overrides=(change for change in ["params.max_depth=3"]),
    )
    slower = compose_config(
        config_dir=tmp_path,
        config_name="baseline",
        overrides=["params.max_depth=3", "params.learning_rate=0.03"],
    )
    assert depth3.config["params"]["max_depth"] == 3
    assert depth3.overrides == ("params.max_depth=3",)
    slower.config["params"]["metric"].append("average_precision")
    assert baseline.config["params"]["max_depth"] == 4
    assert depth3.config["params"]["learning_rate"] == 0.05
    assert baseline.config["params"]["metric"] == ["auc", "binary_logloss"]
    assert path.read_text() == text


def test_save_derived_config_preserves_baseline_and_existing_files(tmp_path):
    pytest.importorskip("hydra")
    pytest.importorskip("yaml")
    path = tmp_path / "baseline.yaml"
    original = "params:\n  max_depth: 4\n  learning_rate: 0.05\n"
    path.write_text(original)
    cfg = compose_config(
        config_dir=tmp_path, config_name="baseline", overrides=["params.max_depth=3"]
    )
    saved = cfg.save(tmp_path / "depth3.yaml")
    assert read_yaml_config(saved)["params"] == {"max_depth": 3, "learning_rate": 0.05}
    with pytest.raises(ExperimentError, match="exist"):
        cfg.save(path)
    assert path.read_text() == original
    with pytest.raises(ExperimentError, match="exist"):
        cfg.save(saved)
