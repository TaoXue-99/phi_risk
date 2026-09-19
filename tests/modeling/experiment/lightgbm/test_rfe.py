import numpy as np
import pytest
from sklearn import config_context

from phl_risk.exceptions import ExperimentError


@pytest.mark.parametrize("routing", [False, True])
def test_rfe_train_only_weighted_native_candidates(exp, config, monkeypatch, routing):
    import lightgbm as lgb

    baseline = exp.run(name="baseline", config=config)
    original = lgb.LGBMClassifier.fit
    sizes = []

    def observed(self, X, y, **kwargs):
        sizes.append(len(y))
        assert len(y) == 360
        np.testing.assert_allclose(kwargs["sample_weight"], np.linspace(0.5, 2, 600)[:360])
        assert "eval_set" not in kwargs
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", observed)
    with config_context(enable_metadata_routing=routing):
        selected = exp.select_features(base_run=baseline.run_id, candidate_counts=[6, 4], step=0.5)
    assert sizes
    assert [r.n_features for r in selected.runs] == [6, 4]
    for run in selected.runs:
        assert run.model.__class__.__name__ == "Booster"
        assert run.features == tuple(f for f in baseline.features if f in run.features)
        assert set(run.eval_history) == {"train", "test"}
    assert len(exp.compare()) == 3
    assert len(selected.compare()) == 3
    assert selected.within_tolerance(tolerance=1).n_features == 4


@pytest.mark.parametrize("counts", [[0], [11], [True], [4, 4], []])
def test_invalid_candidates(exp, config, counts):
    baseline = exp.run(name="baseline", config=config)
    with pytest.raises(ExperimentError):
        exp.select_features(base_run=baseline.run_id, candidate_counts=counts)


def test_feature_indexed_constraints_rejected_before_rfe(exp, config):
    config["model"]["params"]["monotone_constraints"] = [1, 0, 0, 0]
    baseline = exp.run(name="constrained", config=config, features=["x0", "x1", "x2", "x3"])
    with pytest.raises(ExperimentError, match="RFE.*feature-indexed"):
        exp.select_features(base_run=baseline.run_id, candidate_counts=[2], step=1)
    assert len(exp.runs) == 1
