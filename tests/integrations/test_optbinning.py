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


def test_persistence_and_solver_guard(binary_data, tmp_path):
    from phl_risk.exceptions import ArtifactError

    X, y = binary_data
    step = OptBinningStep(["x"], special_codes=[-999], max_n_prebins=5).fit(X, y)
    path = tmp_path / "bins.joblib"
    step.save(path)
    pd.testing.assert_frame_equal(step.transform(X), OptBinningStep.load(path).transform(X))
    # The safety check inspects the fitted backend, not a mutable constructor dictionary.
    step.binning_process_.get_binned_variable("x").solver = "mip"
    with pytest.raises(ArtifactError, match="mip"):
        step.save(path)


def test_selected_feature_equivalence_and_noncart_weight_guard(binary_data):
    from phl_risk.exceptions import DataPrepError

    X, y = binary_data
    kwargs = dict(
        categorical_variables=["c"],
        special_codes=[-999],
        selection_criteria={"iv": {"strategy": "highest", "top": 1}},
    )
    raw = optbinning.BinningProcess(["x", "c"], **kwargs).fit(X[["x", "c"]], y)
    step = OptBinningStep(
        ["x", "c"],
        categorical_columns=["c"],
        special_codes=[-999],
        selection_criteria=kwargs["selection_criteria"],
    ).fit(X, y)
    selected = list(raw.get_support(names=True))
    assert step.selected_columns_ == tuple(selected)
    pd.testing.assert_frame_equal(step.transform(X)[selected], raw.transform(X[["x", "c"]]))
    assert set(step.transform(X)) == set(selected) | {"keep"}
    with pytest.raises(DataPrepError, match="cart"):
        OptBinningStep(["x"], binning_fit_params={"x": {"prebinning_method": "quantile"}}).fit(
            X, y, np.ones(len(X))
        )
