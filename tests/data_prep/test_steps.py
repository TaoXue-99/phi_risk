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
