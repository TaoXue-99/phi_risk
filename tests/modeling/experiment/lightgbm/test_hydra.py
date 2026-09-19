import subprocess
import sys

import pytest

from phl_risk.exceptions import OptionalDependencyError
from phl_risk.modeling.experiment.lightgbm import compose_lightgbm_config
from phl_risk.modeling.experiment.lightgbm._utils import require


def test_real_hydra_composition(tmp_path):
    pytest.importorskip("hydra")
    (tmp_path / "baseline.yaml").write_text(
        "model:\n  params:\n    max_depth: 3\n    seed: 2026\n"
        "train:\n  num_boost_round: ${model.params.seed}\n"
    )
    before = __import__("os").getcwd()
    resolved = compose_lightgbm_config(
        config_dir=tmp_path, config_name="baseline", overrides=["model.params.max_depth=2"]
    )
    assert resolved.config["model"]["params"]["max_depth"] == 2
    assert resolved.config["train"]["num_boost_round"] == 2026
    assert resolved.overrides == ("model.params.max_depth=2",)
    assert __import__("os").getcwd() == before


@pytest.mark.parametrize("module,extra", [("lightgbm", "lightgbm"), ("hydra", "hydra")])
def test_absent_optional_dependency(monkeypatch, module, extra):
    import importlib

    original = importlib.import_module

    def blocked(name, *args, **kwargs):
        if name == module:
            raise ImportError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(OptionalDependencyError, match=extra):
        require(module, extra)


def test_declaration_stays_backend_free():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import sys
sys.path.insert(0,'src')
from phl_risk.modeling import ModelPlan, DataPlan
assert not {'lightgbm','hydra','pandas','sklearn'}.intersection(sys.modules)
""",
        ],
        check=True,
    )


def test_public_compose_missing_hydra(tmp_path, monkeypatch):
    import importlib

    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "hydra":
            raise ImportError("missing optional Hydra")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(OptionalDependencyError, match=r"phl-risk\[hydra\]"):
        compose_lightgbm_config(config_dir=tmp_path, config_name="baseline")
