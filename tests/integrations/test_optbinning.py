import joblib
import numpy as np
import pandas as pd
import pytest

from phl_risk.data_prep.integrations import OptBinningStep

optbinning = pytest.importorskip("optbinning")


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=400)
    X = pd.DataFrame({"x": x, "c": np.where(x > 0, "a", "b"), "keep": np.arange(400)})
    y = (x + rng.normal(size=400) > 0).astype(int)
    X.loc[0:9, "x"] = -999
    X.loc[10:19, "x"] = np.nan
    return X, y


@pytest.mark.parametrize("metric", ["woe", "event_rate", "indices", "bins"])
def test_backend_equivalence_missing_special_categories_and_frozen_state(binary_data, metric):
    X, y = binary_data
    kwargs = dict(categorical_variables=["c"], special_codes=[-999], max_n_prebins=5)
    raw = optbinning.BinningProcess(["x", "c"], **kwargs).fit(X[["x", "c"]], y)
    step = OptBinningStep(
        ["x", "c"], categorical_columns=["c"], special_codes=[-999], max_n_prebins=5, metric=metric
    ).fit(X, y)
    expected = raw.transform(X[["x", "c"]], metric=metric)
    before = joblib.hash(step)
    actual = step.transform(X)
    pd.testing.assert_frame_equal(actual[["x", "c"]], expected)
    pd.testing.assert_frame_equal(step.summary(), raw.summary())
    pd.testing.assert_frame_equal(
        step.binning_table("x"), raw.get_binned_variable("x").binning_table.build()
    )
    audit = step.run(X).audit
    assert audit.details["x"]["input_missing"] == 10
    assert audit.details["x"]["special_count"] == 10
    step.transform(X.iloc[:30])
    assert joblib.hash(step) == before


def test_weights_per_variable_params_and_append(binary_data):
    X, y = binary_data
    weights = np.linspace(0.5, 2, len(X))
    fit_params = {"x": {"max_n_bins": 3}}
    trans_params = {"x": {"metric_missing": "empirical", "metric_special": "empirical"}}
    raw = optbinning.BinningProcess(
        ["x"],
        special_codes=[-999],
        binning_fit_params=fit_params,
        binning_transform_params=trans_params,
        n_jobs=1,
    ).fit(X[["x"]], y, sample_weight=weights)
    step = OptBinningStep(
        ["x"],
        special_codes=[-999],
        binning_fit_params=fit_params,
        binning_transform_params=trans_params,
        n_jobs=1,
        output="append",
    ).fit(X, y, weights)
    np.testing.assert_allclose(step.transform(X)[["x__woe"]], raw.transform(X[["x"]], metric="woe"))
