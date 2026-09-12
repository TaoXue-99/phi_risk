import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from phl_risk.data_prep import (
    BasePrepStep,
    DataPrep,
    FittedPrepStep,
    KBinsStep,
    LogitTransform,
    MissingImputer,
    PrepAudit,
    StatelessPrepStep,
    ToDatetime,
    ToNumeric,
    ValueMapper,
)
from phl_risk.exceptions import DataPrepError, NotFittedError


class ChangingSchema(StatelessPrepStep):
    def _transform(self, X):
        return X.rename(columns={"x": "other"}) if X.x.lt(0).any() else X


class BrokenRows(StatelessPrepStep):
    def __init__(self, mode):
        self.mode = mode

    def _transform(self, X):
        if self.mode == "drop":
            return X.iloc[1:]
        if self.mode == "add":
            return pd.concat([X, X.iloc[:1]])
        if self.mode == "reorder":
            return X.iloc[::-1]
        if self.mode == "index":
            return X.reset_index(drop=True)
        return X.to_numpy()


class FittedCenter(FittedPrepStep):
    def _fit(self, X, y=None, sample_weight=None):
        self.center_ = X.x.mean()

    def _transform(self, X):
        if X.x.lt(0).any():
            return X.iloc[1:]
        if X.x.gt(100).any():
            return X.rename(columns={"x": "new"})
        return X.assign(x=X.x - self.center_)


def test_types_no_fake_fit_and_parameter_clone():
    assert not hasattr(BasePrepStep, "fit")
    assert not hasattr(BasePrepStep, "_fit")
    assert not hasattr(BasePrepStep, "requires_fit")
    X = pd.DataFrame({"x": ["1", "2"]})
    for step in [
        ToNumeric(["x"]),
        ValueMapper("x", {"1": 1, "2": 2}),
        ToDatetime(["x"], format="%d"),
    ]:
        assert isinstance(step, StatelessPrepStep)
        assert not hasattr(step, "fit")
        params = joblib.hash(step.get_params())
        step.transform(X)
        step.fit_transform(X)
        step.run(X)
        assert not any(name.endswith("_") for name in vars(step))
        assert not hasattr(step, "mapping_")
        assert joblib.hash(step.get_params()) == params
        assert joblib.hash(clone(step)) == joblib.hash(step)
    assert isinstance(MissingImputer(["x"]), FittedPrepStep)
    assert isinstance(KBinsStep(["x"]), FittedPrepStep)


def test_mixed_sequential_lifecycle_and_no_stateless_fit(monkeypatch):
    X = pd.DataFrame(
        {"income": [str(i) for i in range(1, 101)], "prob": [str(i / 101) for i in range(1, 101)]}
    )
    X.loc[0, ["income", "prob"]] = None

    def forbidden(*args, **kwargs):
        raise AssertionError("Stateless fit_transform must not be dispatched")

    monkeypatch.setattr(StatelessPrepStep, "fit_transform", forbidden)
    prep = DataPrep(
        [
            ToNumeric(["income", "prob"]),
            LogitTransform("prob", "prob_logit"),
            MissingImputer(["income", "prob_logit"]),
            KBinsStep(["income"], n_bins=2),
        ]
    )
    prep.fit(X)
    assert not any(k.endswith("_") for _, s in prep.steps_[:2] for k in vars(s))
    probabilities = np.arange(2, 101) / 101
    expected = np.median(np.log(probabilities) - np.log1p(-probabilities))
    np.testing.assert_allclose(prep.steps_[2][1].imputer_.statistics_, [51, expected])
    assert prep.steps_[3][1].transformer_.bin_edges_[0][0] == 2
    before = joblib.hash(prep)
    result = prep.run(X)
    assert joblib.hash(prep) == before
    assert [a.kind for a in result.audits] == ["stateless", "stateless", "fitted", "fitted"]
    assert result.audits[1].output_columns == ("prob_logit",)
    assert result.audit_frame().kind.tolist() == [a.kind for a in result.audits]
    assert prep.get_feature_names_out().tolist() == ["income", "prob", "prob_logit"]


@pytest.mark.parametrize("operation", ["transform", "run"])
@pytest.mark.parametrize("change", ["missing", "extra", "reorder"])
def test_composite_schema_contract(operation, change):
    X = pd.DataFrame({"x": ["1"], "y": ["2"]})
    prep = DataPrep([ToNumeric(["x"])]).fit(X)
    changed = {"missing": X[["x"]], "extra": X.assign(z=1), "reorder": X[["y", "x"]]}[change]
    with pytest.raises(DataPrepError, match="columns/order"):
        getattr(prep, operation)(changed)


@pytest.mark.parametrize("operation", ["transform", "run"])
def test_stateless_output_schema_cannot_drift(operation):
    X = pd.DataFrame({"x": [1.0]})
    prep = DataPrep([ChangingSchema()]).fit(X)
    with pytest.raises(DataPrepError, match="output columns/order"):
        getattr(prep, operation)(-X)


@pytest.mark.parametrize("mode", ["drop", "add", "reorder", "index", "array"])
def test_row_and_dataframe_contract(mode):
    X = pd.DataFrame({"x": [1, 2, 3]}, index=[4, 5, 9])
    original = X.copy()
    with pytest.raises(DataPrepError):
        BrokenRows(mode).run(X)
    pd.testing.assert_frame_equal(X, original)


def test_fitted_schema_check_atomic_refit_and_state_freeze():
    X = pd.DataFrame({"x": [1.0, 3.0]})
    step = FittedCenter()
    with pytest.raises(NotFittedError):
        step.transform(X)
    step.fit(X)
    before = joblib.hash(step)
    with pytest.raises(DataPrepError, match="preserve"):
        step.fit(-X)
    assert joblib.hash(step) == before
    with pytest.raises(DataPrepError, match="output schema"):
        step.transform(X + 200)
    pd.testing.assert_frame_equal(step.transform(X), X - 2)
    step.run(X + 10)
    assert joblib.hash(step) == before


def test_mapping_configuration_isolated_by_composite_and_persistence(tmp_path):
    mapping = {"a": 1, "b": 2}
    step = ValueMapper("x", mapping)
    X = pd.DataFrame({"x": ["a", "b"]})
    prep = DataPrep([step]).fit(X)
    mapping["a"] = 99
    assert step.transform(X).x.iloc[0] == 99  # Standalone configuration remains configurable.
    assert prep.transform(X).x.iloc[0] == 1  # Fitted plan owns a clone of that configuration.
    assert not hasattr(prep.steps_[0][1], "mapping_")
    path = tmp_path / "stateless.joblib"
    prep.save(path)
    loaded = DataPrep.load(path)
    pd.testing.assert_frame_equal(loaded.transform(X), prep.transform(X))
    audit = loaded.run(X).audits[0]
    old_shape = PrepAudit(
        audit.step,
        audit.input_columns,
        audit.output_columns,
        audit.before,
        audit.after,
        audit.details,
    )
    assert old_shape.kind is None


def test_legacy_composite_requires_explicit_refit():
    X = pd.DataFrame({"x": ["1", "2"]})
    prep = DataPrep([ToNumeric(["x"])]).fit(X)
    del prep.step_feature_names_out_
    with pytest.raises(DataPrepError, match="Legacy.*refit"):
        prep.transform(X)
    assert prep.fit_transform(X).x.tolist() == [1, 2]
