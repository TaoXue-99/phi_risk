import joblib
import numpy as np
import pandas as pd
import pytest

from phl_risk.data_prep import Clip, DataPrep, LogitTransform, LogTransform, StatelessPrepStep
from phl_risk.exceptions import DataPrepError


def test_clip_counts_nulls_and_duplicate_index():
    X = pd.DataFrame({"x": [-1.0, 0.0, 0.5, 1.0, 2.0, np.nan], "keep": range(6)}, index=[3] * 6)
    original = X.copy()
    step = Clip(["x"], 0, 1)
    before = joblib.hash(step)
    result = step.run(X)
    np.testing.assert_allclose(result.data.x, [0, 0, 0.5, 1, 1, np.nan])
    assert result.audit.details["x"]["clipped_count"] == 2
    assert result.audit.kind == "stateless"
    assert result.data.index.equals(X.index)
    assert joblib.hash(step) == before
    pd.testing.assert_frame_equal(X, original)


def test_log_and_generated_column_audit():
    X = pd.DataFrame({"x": [1.0, np.e, np.nan]}, index=[7, 7, 9])
    step = LogTransform("x", "logged")
    result = step.run(X)
    np.testing.assert_allclose(result.data.logged, [0.0, 1.0, np.nan])
    assert result.audit.before.columns == ("x",)
    assert result.audit.after.columns == ("logged",)
    assert result.audit.details["transformed_count"] == 2
    assert result.data.index.equals(X.index)
    pd.testing.assert_series_equal(result.data.x, X.x)
    assert "logged" not in X


def test_logit_boundaries_outliers_null_and_no_learning():
    X = pd.DataFrame({"p": [-np.inf, -1, 0, 0.2, 0.5, 1, 2, np.inf, np.nan]})
    step = LogitTransform("p", "logit", eps=0.01)
    result = step.run(X)
    expected_p = np.clip(X.p, 0.01, 0.99)
    np.testing.assert_allclose(result.data.logit, np.log(expected_p / (1 - expected_p)))
    assert result.audit.details["clipped_count"] == 6
    assert result.audit.after.missing_count["logit"] == 1
    assert isinstance(step, StatelessPrepStep) and not hasattr(step, "_is_fitted_")


@pytest.mark.parametrize("step", [Clip(["x"], 0, 1), LogTransform("x"), LogitTransform("x")])
def test_empty_and_nullable_inputs(step):
    empty = pd.DataFrame({"x": pd.Series(dtype=float)})
    assert step.run(empty).data.empty
    X = pd.DataFrame({"x": pd.Series([0.5, None], dtype="Float64")})
    output = step.transform(X)
    assert pd.isna(output.x.iloc[1])
    assert output.index.equals(X.index)


@pytest.mark.parametrize(
    "step",
    [
        Clip(["x"], 2, 1),
        Clip(["x"], np.nan),
        LogitTransform("x", eps=0),
        LogitTransform("x", eps=0.5),
        LogitTransform("x", eps=1e-30),
        LogitTransform("x", eps=None),
        LogitTransform("x", eps=True),
        LogitTransform("x", eps=np.inf),
        LogTransform("x", ""),
    ],
)
def test_invalid_configuration(step):
    with pytest.raises(DataPrepError):
        step.transform(pd.DataFrame({"x": [0.5]}))


@pytest.mark.parametrize("step", [LogTransform("x", "keep"), LogitTransform("x", "keep")])
def test_output_collision(step):
    with pytest.raises(DataPrepError, match="already exists"):
        step.transform(pd.DataFrame({"x": [0.5], "keep": [1]}))


@pytest.mark.parametrize("value", [0.0, -1.0, -np.inf])
def test_log_rejects_nonpositive(value):
    with pytest.raises(DataPrepError, match="positive"):
        LogTransform("x").transform(pd.DataFrame({"x": [value]}))


@pytest.mark.parametrize("step", [Clip(["x"]), LogTransform("x"), LogitTransform("x")])
def test_numeric_input_is_explicit(step):
    for values in [["1"], [True], [1 + 2j]]:
        with pytest.raises(DataPrepError, match="real numeric"):
            step.transform(pd.DataFrame({"x": values}))


def test_numeric_only_plan_roundtrip(tmp_path):
    X = pd.DataFrame({"x": [0.2, 0.5, 0.8]})
    prep = DataPrep(
        [Clip(["x"], 0.01, 0.99), LogitTransform("x", "odds"), LogTransform("x", "log")]
    ).fit(X)
    before = joblib.hash(prep)
    output = prep.run(X)
    assert joblib.hash(prep) == before
    path = tmp_path / "numeric.joblib"
    prep.save(path)
    pd.testing.assert_frame_equal(DataPrep.load(path).transform(X), output.data)


def test_clip_preserves_large_integer_values_and_nullable_fractional_bounds():
    X = pd.DataFrame({"x": pd.Series([2**60 + 1, None], dtype="Int64")})
    pd.testing.assert_frame_equal(Clip(["x"], lower=0).transform(X), X)
    small = pd.DataFrame({"x": pd.Series([0, 2, None], dtype="Int64")})
    np.testing.assert_allclose(Clip(["x"], 0.5, 1.5).transform(small).x, [0.5, 1.5, np.nan])
