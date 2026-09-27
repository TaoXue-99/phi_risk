import numpy as np
import pandas as pd
import pytest

from phl_risk.modeling.experiment import Experiment


@pytest.fixture
def native():
    pytest.importorskip("lightgbm")
    from phl_risk.modeling.experiment.adapters.lightgbm import enable_lightgbm_compatibility

    return enable_lightgbm_compatibility()


@pytest.fixture
def data():
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame(rng.normal(size=(600, 10)), columns=[f"x{i}" for i in range(10)])
    frame["y"] = (frame.x0 + frame.x1 > 0).astype(int)
    frame["w"] = rng.uniform(0.5, 2, len(frame))
    return frame


@pytest.fixture
def space(tmp_path):
    return Experiment(root=tmp_path / "experiments").initialize(
        method="lgb", objective="binary", metric="auc"
    )


@pytest.fixture
def record_facts():
    return {
        "model": {"family": "test", "backend": "example", "backend_version": "1"},
        "input": {"features": ["x"]},
        "configuration": {"supplied": {}, "resolved": {}},
        "environment": {},
    }
