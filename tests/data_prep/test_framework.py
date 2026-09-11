import joblib
import pandas as pd
import pytest
from sklearn.base import clone

from phl_risk.data_prep import BasePrepStep, DataPrep
from phl_risk.exceptions import DataPrepError, NotFittedError


class DummyStatelessPrep(BasePrepStep):
    requires_fit = False

    def _transform(self, X):
        X["x"] = X["x"] * 2
        return X


class DummyFittedPrep(BasePrepStep):
    def _fit(self, X, y=None, sample_weight=None):
        self.mean_ = X.x.mean()

    def _transform(self, X):
        X["x"] = X.x - self.mean_
        return X


def test_sequential_fit_state_results_and_plugins():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[1, 1, 7])
    original = X.copy()
    stateless = DummyStatelessPrep()
    assert stateless.transform(X).x.tolist() == [2, 4, 6]
    prep = DataPrep([stateless, DummyFittedPrep()])
    with pytest.raises(NotFittedError):
        prep.transform(X)
    out = prep.fit_transform(X)
    assert prep.steps_[1][1].mean_ == 4
    assert out.x.tolist() == [-2, 0, 2]
    assert not hasattr(stateless, "feature_names_in_")
    before = joblib.hash(prep)
    result = prep.run(X + 100)
    prep.transform(X + 1000)
    assert joblib.hash(prep) == before
    pd.testing.assert_frame_equal(original, X)
    assert len(result.audits) == 2
    data = result.data
    data.iloc[0, 0] = -999
    assert result.data.iloc[0, 0] != -999
    assert prep.get_feature_names_out().tolist() == ["x"]
    assert not hasattr(clone(prep), "steps_")
    with pytest.raises(DataPrepError):
        prep.transform(X.rename(columns={"x": "z"}))
