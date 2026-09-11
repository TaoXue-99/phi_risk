from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    PSI,
    BinDimension,
    Col,
    ColumnDimension,
    ComparativeMeasure,
    ComputePolicy,
    Cube,
    CubeResult,
    MissingPolicy,
    QuantileBinner,
)
from phl_risk.analysis._comparison_types import ComparativeCalculation, ComparisonOutput
from phl_risk.analysis._nodes import ComparativeNode, MeasureSpec
from phl_risk.exceptions import EngineError, InvalidMetricError, MeasureError, NotFittedError


def naive(ref, cur, field, dimensions, bins=4, epsilon=1e-8):
    """Independent quantiles + pandas.cut + per-group value_counts oracle."""
    edges = np.unique(np.quantile(ref[field].dropna(), np.linspace(0, 1, bins + 1)))
    edges = np.r_[-np.inf, edges[1:-1], np.inf]
    size = len(edges) - 1
    a = ref.assign(_bin=pd.cut(ref[field], edges, labels=False, include_lowest=True).fillna(size))
    b = cur.assign(_bin=pd.cut(cur[field], edges, labels=False, include_lowest=True).fillna(size))
    has_missing = ref[field].isna().any() or cur[field].isna().any()
    domain = range(size + int(has_missing))
    if dimensions:
        keys = (
            pd.concat([ref[dimensions], cur[dimensions]])
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    else:
        keys = [()]
    results = []
    for key in keys:
        left, right = a, b
        for column, value in zip(dimensions, key):
            left, right = left[left[column] == value], right[right[column] == value]
        if not len(left) or not len(right):
            results.append(np.nan)
            continue
        p = left._bin.value_counts().reindex(domain, fill_value=0).to_numpy() / len(left)
        q = right._bin.value_counts().reindex(domain, fill_value=0).to_numpy() / len(right)
        p, q = np.maximum(p, epsilon), np.maximum(q, epsilon)
        p, q = p / p.sum(), q / q.sum()
        results.append(sum((y - x) * np.log(y / x) for x, y in zip(p, q)))
    return results


@pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4])
def test_n_dimensional_math_and_layout(ndim):
    rng = np.random.default_rng(12)
    dimensions = [f"d{i}" for i in range(ndim)]
    ref = pd.DataFrame(
        {"x": rng.normal(size=400), **{d: rng.integers(0, 2, 400) for d in dimensions}}
    )
    cur = pd.DataFrame(
        {"x": rng.normal(1, size=230), **{d: rng.integers(0, 2, 230) for d in dimensions}}
    )
    ref.loc[0, "x"] = np.nan
    cur.loc[2, "x"] = np.nan
    original = ref.copy(deep=True)
    cube = Cube(dimensions, [PSI("x", QuantileBinner(4))])
    result = cube.compute_comparison(ref, cur)
    assert isinstance(result, CubeResult)
    np.testing.assert_allclose(result.data_.value, naive(ref, cur, "x", dimensions))
    assert list(result.data_.columns) == dimensions + ["metric", "value"]
    assert result.layout().size == 2**ndim
    assert result.metadata_["group_encoding_passes"] == 1
    pd.testing.assert_frame_equal(original, ref)


def test_identical_shifted_and_unequal_sizes():
    ref = pd.DataFrame({"x": np.arange(20.0)})
    cube = Cube(measures=[PSI("x")])
    assert cube.compute_comparison(ref, pd.concat([ref] * 3)).data_.value.iloc[0] == 0
    assert cube.compute_comparison(ref, ref + 100).data_.value.iloc[0] > 1
    assert not cube.measures[0].binner.is_fitted


@pytest.mark.parametrize("fields", [3, 100])
def test_batch_fields_share_encoding(monkeypatch, fields):
    import phl_risk.analysis.engines._comparison as implementation

    original = implementation.encode_groups
    calls = []

    def spy(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(implementation, "encode_groups", spy)
    rng = np.random.default_rng(6)
    ref = pd.DataFrame(rng.normal(size=(100, fields)), columns=[f"x{i}" for i in range(fields)])
    ref["group"] = np.arange(100) % 2
    cube = Cube(["group"], [PSI(f"x{i}") for i in range(fields)])
    result = cube.compute_comparison(ref, ref)
    assert len(calls) == 1
    assert result.shape == (2, fields)
    assert (result.data_.value == 0).all()
    assert result.axes[-1].values == tuple(f"psi__x{i}" for i in range(fields))


def test_union_groups_categories_and_missing_dimensions():
    ref = pd.DataFrame(
        {
            "g": pd.Categorical(["b", None, "a"], categories=["b", "a", "unused"]),
            "x": ["A", "B", "C"],
        }
    )
    cur = pd.DataFrame({"g": ["b", None, "c"], "x": ["D", "B", "A"]})
    cube = Cube(
        ["g"],
        [PSI("x", binner=None)],
        policy=ComputePolicy(missing=MissingPolicy(dimension="keep")),
    )
    result = cube.compute_comparison(ref, cur)
    assert result.axes[0].values == ("b", "a", "unused", "c", None)
    data = result.data_.set_index("g").value
    assert data.loc["b"] > 0
    assert data[data.index.isna()].iloc[0] == 0
    assert np.isnan(data.loc["a"]) and np.isnan(data.loc["c"])
    assert np.isnan(result.layout().loc["unused"].iloc[0])


@pytest.mark.parametrize("missing", ["bucket", "drop"])
def test_field_missing_policy(missing):
    ref = pd.DataFrame({"x": [0.0, 1.0, np.nan, np.nan]})
    cur = pd.DataFrame({"x": [0.0, 1.0]})
    result = Cube(measures=[PSI("x", missing=missing)]).compute_comparison(ref, cur)
    value = result.data_.value.iloc[0]
    assert (value > 0) if missing == "bucket" else (value == 0)


@pytest.mark.parametrize("kind", ["empty", "missing"])
@pytest.mark.parametrize("policy", ["nan", "warn", "raise"])
def test_unavailable_numeric_reference(kind, policy):
    ref = pd.DataFrame({"x": pd.Series([], dtype=float) if kind == "empty" else [np.nan]})
    cur = pd.DataFrame({"x": [1.0]})
    cube = Cube(measures=[PSI("x")], policy=ComputePolicy(on_invalid=policy))
    if policy == "raise":
        with pytest.raises(InvalidMetricError):
            cube.compute_comparison(ref, cur)
    elif policy == "warn":
        with pytest.warns(RuntimeWarning):
            cube.compute_comparison(ref, cur)
    else:
        assert np.isnan(cube.compute_comparison(ref, cur).data_.value.iloc[0])


def test_empty_both_and_all_missing_categorical():
    empty = pd.DataFrame({"g": pd.Series([], dtype=str), "x": pd.Series([], dtype=float)})
    assert Cube(["g"], [PSI("x")]).compute_comparison(empty, empty).data_.empty
    missing = pd.DataFrame({"x": [None, None]})
    assert (
        Cube(measures=[PSI("x", binner=None)])
        .compute_comparison(missing, missing)
        .data_.value.iloc[0]
        == 0
    )
    assert np.isnan(
        Cube(measures=[PSI("x", binner=None, missing="drop")])
        .compute_comparison(missing, missing)
        .data_.value.iloc[0]
    )


def test_min_samples_filters_and_duplicate_index():
    ref = pd.DataFrame({"g": [0, 0, 1], "x": [0.0, 1.0, 2.0]}, index=[0, 0, 0])
    cube = Cube(["g"], [PSI("x", min_samples=2)], filters=[Col("x") >= 0])
    result = cube.compute_comparison(ref, ref).data_.set_index("g").value
    assert result.loc[0] == 0 and np.isnan(result.loc[1])
    with pytest.raises(InvalidMetricError, match="min_samples"):
        Cube(
            ["g"], [PSI("x", min_samples=2)], policy=ComputePolicy(on_invalid="raise")
        ).compute_comparison(ref, ref)


def test_dimension_fit_lifecycle_and_renaming():
    ref = pd.DataFrame({"x": np.arange(10.0), "g": np.arange(10.0)})
    cube = Cube([BinDimension("g", QuantileBinner(2), name="group")], [PSI("x", name="stability")])
    with pytest.raises(NotFittedError):
        cube.compute_comparison(ref, ref)
    cube.fit(ref)
    edges = cube.dimensions_[0].transformer.bin_edges_.copy()
    result = cube.compute_comparison(ref, ref)
    assert list(result.data_.columns) == ["group", "metric", "value"]
    assert "stability" in result.layout(bin_labels="interval")
    np.testing.assert_array_equal(edges, cube.dimensions_[0].transformer.bin_edges_)
    renamed = Cube([ColumnDimension("g", "other")], [PSI("x")]).compute_comparison(ref, ref)
    assert renamed.axes[0].name == "other"


def test_validation():
    with pytest.raises(MeasureError, match="Duplicate"):
        Cube(measures=[PSI("x"), PSI("x")]).plan()
    with pytest.raises(EngineError, match="Missing"):
        Cube(measures=[PSI("x")]).compute_comparison(pd.DataFrame(), pd.DataFrame())
    for kwargs in [{"missing": "bad"}, {"min_samples": 0}, {"binner": 10}]:
        with pytest.raises(MeasureError):
            PSI("x", **kwargs)


@dataclass(frozen=True)
class RowDifferenceCalculation(ComparativeCalculation):
    """Test-only non-distribution metric proves engine does not dispatch on PSI."""

    def required_columns(self):
        return ()

    def evaluate(self, reference, current, groups, *, backend, on_invalid):
        a = np.bincount(groups.reference_codes, minlength=groups.size)
        b = np.bincount(groups.current_codes, minlength=groups.size)
        return ComparisonOutput((b - a).astype(float))


@dataclass(frozen=True)
class RowDifference(ComparativeMeasure):
    def compile(self, context=None):
        return MeasureSpec("rows_changed", ComparativeNode(RowDifferenceCalculation()))


def test_non_psi_extension_uses_same_engine():
    ref, cur = pd.DataFrame({"g": ["a", "a", "b"]}), pd.DataFrame({"g": ["a", "b"]})
    result = Cube(["g"], [RowDifference()]).compute_comparison(ref, cur)
    assert result.data_.value.tolist() == [-1, 0]


def test_empty_current_and_missing_numeric_group():
    ref = pd.DataFrame({"g": [0, 0, 1], "x": [0.0, 1.0, np.nan]})
    cube = Cube(["g"], [PSI("x")])
    values = cube.compute_comparison(ref, ref).data_.value
    np.testing.assert_allclose(values, [0, 0])
    assert cube.compute_comparison(ref, ref.iloc[:0]).data_.value.isna().all()


def test_dimension_drop_and_numeric_outliers():
    ref = pd.DataFrame({"g": [0, 0, None], "x": [0.0, 1.0, 999.0]})
    cur = pd.DataFrame({"g": [0, 0, None], "x": [-np.inf, np.inf, np.nan]})
    cube = Cube(["g"], [PSI("x", QuantileBinner(2))])
    result = cube.compute_comparison(ref, cur)
    assert result.data_.value.iloc[0] == 0
    assert result.metadata_["reference_analyzed_rows"] == 2
    assert result.metadata_["measures"]["psi__x"]["transform"]["reference_range"] == (0.0, 1.0)


def test_filter_and_dimension_processing_independent_of_field_count():
    calls = []

    def predicate(frame):
        calls.append(len(frame))
        return frame.x > 0

    frame = pd.DataFrame({"g": [0, 0, 1], "x": [0.0, 1.0, 2.0]})
    cube = Cube(["g"], [PSI("x", name=f"p{i}") for i in range(10)], filters=[predicate])
    result = cube.compute_comparison(frame, frame)
    assert calls == [3, 3]
    assert (result.data_.value == 0).all()


def test_comparison_declines_totals_and_weight():
    from phl_risk.analysis import AnalysisContext, PandasEngine

    frame = pd.DataFrame({"x": [0.0, 1.0]})
    cube = Cube(measures=[PSI("x")])
    with pytest.raises(EngineError, match="totals"):
        PandasEngine().execute_comparison(cube.plan(totals=True), frame, frame)
    with pytest.raises(MeasureError, match="unweighted"):
        cube.compute_comparison(frame, frame, context=AnalysisContext(weight="w"))
