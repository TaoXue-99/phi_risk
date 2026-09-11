import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, RobustScaler, StandardScaler

from phl_risk.data_prep import DataPrep
from phl_risk.data_prep.steps import (
    KBinsStep,
    MissingImputer,
    SklearnStep,
    ToDatetime,
    ToNumeric,
    ValueMapper,
)
from phl_risk.exceptions import DataPrepError


def test_conversion_audits_and_mapping():
    X = pd.DataFrame({"x": ["1000", "1000 PHP", "unknown", None]})
    run = ToNumeric(["x"], errors="coerce").run(X)
    assert run.audit.details["x"]["new_null"] == 2
    assert X.x.iloc[1] == "1000 PHP"
    dates = pd.DataFrame({"dt": ["2026-01-01", "bad"]})
    assert (
        ToDatetime(["dt"], "coerce", "%Y-%m-%d").run(dates).audit.details["dt"]["new_nat_count"]
        == 1
    )
    with pytest.raises(DataPrepError):
        ValueMapper("x", {"1000": 1}).transform(X)
    out = ValueMapper("x", {"1000": 1}, "value", -1).run(X)
    assert out.audit.details["unknown_count"] == 2
    assert out.data.x.iloc[1] == -1


def test_imputer_reference_equivalence_and_freeze():
    X = pd.DataFrame({"x": [0.0, 100.0, 200.0, np.nan]})
    oot = pd.DataFrame({"x": [10000.0, np.nan]}, index=[9, 9])
    raw = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
    step = MissingImputer(["x"]).fit(X)
    np.testing.assert_allclose(step.transform(oot), raw.transform(oot))
    assert step.transform(oot).x.iloc[1] == 100
    before = joblib.hash(step)
    assert step.run(oot).audit.details["x"]["imputed_count"] == 1
    assert joblib.hash(step) == before
    assert step.imputer_ is step.transformer_


@pytest.mark.parametrize("transformer", [RobustScaler(), StandardScaler()])
def test_sklearn_generic_equivalence(transformer):
    X = pd.DataFrame({"x": [1.0, 3.0, 10.0], "keep": [3, 2, 1]})
    wrapper = SklearnStep(transformer, ["x"]).fit(X)
    raw = type(transformer)().fit(X[["x"]])
    np.testing.assert_allclose(wrapper.transform(X)[["x"]], raw.transform(X[["x"]]))
    assert not hasattr(transformer, "n_features_in_")


def test_kbins_weights_and_onehot_names():
    X = pd.DataFrame({"x": np.arange(100.0)})
    step = KBinsStep(["x"], n_bins=5).fit(X)
    raw = KBinsDiscretizer(
        n_bins=5, encode="ordinal", random_state=0, quantile_method="averaged_inverted_cdf"
    ).fit(X)
    np.testing.assert_array_equal(step.transform(X), raw.transform(X))
    cats = pd.DataFrame({"c": ["a", "b", "a"], "keep": [1, 2, 3]}, index=[4, 4, 8])
    encoder = SklearnStep(OneHotEncoder(handle_unknown="ignore"), ["c"]).fit(cats)
    result = encoder.run(cats)
    assert list(result.data) == ["c_a", "c_b", "keep"]
    assert isinstance(result.data.c_a.dtype, pd.SparseDtype)
    with pytest.raises(DataPrepError, match="sample_weight"):
        SklearnStep(RobustScaler(), ["x"]).fit(X, sample_weight=np.ones(100))


def test_sequential_numeric_and_imputation():
    X = pd.DataFrame({"x": ["10", "20", None]})
    prep = DataPrep([ToNumeric(["x"], "coerce"), MissingImputer(["x"])]).fit(X)
    assert prep.transform(pd.DataFrame({"x": ["bad"]})).x.iloc[0] == 15


def test_weighted_scaler_and_imputer_empty_feature_indicator():
    X = pd.DataFrame({"x": [1.0, 2.0, 10.0]})
    weights = np.array([1.0, 1.0, 8.0])
    raw = StandardScaler().fit(X, sample_weight=weights)
    step = SklearnStep(StandardScaler(), ["x"]).fit(X, sample_weight=weights)
    np.testing.assert_allclose(step.transform(X), raw.transform(X))
    missing = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, np.nan]})
    step = MissingImputer(["x", "y"], add_indicator=True).fit(missing)
    raw = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(
        missing
    )
    np.testing.assert_allclose(step.transform(missing), raw.transform(missing))
    assert list(step.transform(missing)) == list(raw.get_feature_names_out())


def test_append_collisions_and_dense_onehot():
    X = pd.DataFrame({"c": ["a", "b"]})
    step = SklearnStep(OneHotEncoder(sparse_output=False), ["c"]).fit(X)
    assert list(step.transform(X)) == ["c_a", "c_b"]
    with pytest.raises(DataPrepError, match="collide"):
        SklearnStep(RobustScaler(), ["x"], output="append").fit(pd.DataFrame({"x": [1.0, 2.0]}))
    append = SklearnStep(RobustScaler(), ["x"], output="append", output_columns=["scaled"])
    assert list(append.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))) == ["x", "scaled"]


def test_imputation_audit_renamed_output_and_nullable_marker():
    X = pd.DataFrame({"x": pd.Series([1.0, None, 3.0], dtype="Float64")})
    step = MissingImputer(["x"], missing_values=pd.NA, output_columns=["filled"]).fit(X)
    result = step.run(X)
    assert result.audit.details["x"]["imputed_count"] == 1
    assert result.audit.details["x"]["imputed_rate"] == pytest.approx(1 / 3)
    assert result.audit.details["x"]["output_column"] == "filled"
