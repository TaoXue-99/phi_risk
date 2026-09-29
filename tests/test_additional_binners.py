import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    PSI,
    BinDimension,
    Count,
    Cube,
    EqualWidthBinner,
    FixedBinner,
)
from phl_risk.exceptions import NotFittedError, TransformError


def test_equal_width_reference_and_current():
    b = EqualWidthBinner(4)
    with pytest.raises(NotFittedError):
        b.transform([1])
    b.fit([0, 1, 2, 100, None, np.inf])
    np.testing.assert_equal(b.bin_edges_, [-np.inf, 25, 50, 75, np.inf])
    x = pd.Series([-np.inf, 0, 25, 26, 50, 75, 101, np.inf, np.nan], index=[2] * 9)
    expected = pd.cut(x, b.bin_edges_, labels=["B1", "B2", "B3", "B4"], include_lowest=True)
    pd.testing.assert_series_equal(b.transform(x), expected)
    edges = b.bin_edges_
    edges[1] = 99
    assert b.bin_edges_[1] == 25
    assert b.metadata()["reference_range"] == (0, 100)


def test_equal_width_extreme_constant_and_atomic_refit():
    b = EqualWidthBinner(2).fit([-1e308, 1e308])
    np.testing.assert_equal(b.bin_edges_, [-np.inf, 0, np.inf])
    b.fit([4, 4])
    assert b.n_bins_ == 1
    assert b.transform([-100, 100]).tolist() == ["B1", "B1"]
    with pytest.raises(TransformError, match="no finite"):
        b.fit([np.nan, np.inf])
    assert b.n_bins_ == 1
    with pytest.raises(TransformError, match="labels"):
        EqualWidthBinner(2, labels=["a", "b"]).fit([1, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 0},
        {"n_bins": True},
        {"n_bins": 1.2},
        {"precision": -1},
        {"labels": ["a", "a"]},
        {"include_lowest": 1},
    ],
)
def test_equal_width_invalid_configuration(kwargs):
    with pytest.raises(TransformError):
        EqualWidthBinner(**kwargs)


@pytest.mark.parametrize("include_lowest", [True, False])
def test_fixed_matches_pandas_and_needs_no_fit(include_lowest):
    b = FixedBinner([0, 10, 30], labels=["low", "high"], include_lowest=include_lowest)
    x = pd.Series([-np.inf, -1, 0, 1, 10, 11, 30, 31, np.inf, np.nan], name="score")
    expected = pd.cut(x, [0, 10, 30], labels=["low", "high"], include_lowest=include_lowest)
    pd.testing.assert_series_equal(b.transform(x), expected)
    assert b.is_fitted
    old = b.bin_edges_
    assert b.fit([100, 200]) is b
    np.testing.assert_equal(old, b.bin_edges_)
    assert b.metadata()["intervals"][0].startswith("[" if include_lowest else "(")


@pytest.mark.parametrize(
    "edges",
    [
        [],
        [0],
        [0, 0, 1],
        [2, 1],
        [0, np.nan, 1],
        [[0, 1], [2, 3]],
        "012",
        [0, np.inf, 3],
        ["a", "b"],
        [False, True],
        [0, 1j],
    ],
)
def test_fixed_bad_edges(edges):
    with pytest.raises(TransformError):
        FixedBinner(edges)


def test_fixed_snapshot_precision_and_infinities():
    edges = [-np.inf, 0.123456789, np.inf]
    b = FixedBinner(edges, precision=2)
    edges[1] = 999
    assert b.bin_edges_[1] == 0.123456789
    assert b.transform([-np.inf, 0.123456789, np.inf]).tolist() == ["B1", "B1", "B2"]
    assert "0.12" in b.metadata()["intervals"][0]
    assert pd.isna(FixedBinner([-np.inf, 0, np.inf], include_lowest=False).transform([-np.inf])[0])


def test_cube_integration_shared_source_layout_totals():
    ref = pd.DataFrame({"a": [0, 25, 50, 100], "b": [10, 30, 60, 90]})
    cube = Cube([BinDimension("b", EqualWidthBinner(2), fit_field="a")], [Count()]).fit(ref)
    result = cube.compute(ref, totals=True)
    assert result.data_["value"].sum() == 4
    assert result.layout(bin_labels="interval", totals=True).shape[0] == 3
    fixed = Cube([BinDimension("b", FixedBinner([0, 20, 100]))], [Count()])
    assert fixed.compute(ref).data_["value"].tolist() == [1, 3]
    assert fixed.fit_compute(ref).data_.equals(fixed.compute(ref).data_)


@pytest.mark.parametrize("binner", [EqualWidthBinner(3), FixedBinner([-np.inf, 1, 2, np.inf])])
def test_psi_uses_new_transformers(binner):
    df = pd.DataFrame({"x": [0, 1, 2, 3, np.nan]})
    result = Cube(measures=[PSI("x", binner=binner)]).compute_comparison(df, df)
    assert result.data_["value"].iloc[0] == pytest.approx(0)


def test_equal_width_psi_invalid_reference_policy():
    ref = pd.DataFrame({"x": [np.nan]})
    cur = pd.DataFrame({"x": [1.0]})
    result = Cube(measures=[PSI("x", binner=EqualWidthBinner())]).compute_comparison(ref, cur)
    assert np.isnan(result.data_["value"].iloc[0])


@pytest.mark.parametrize("include_lowest", [True, False])
def test_three_strategies_share_assignment_and_metadata(include_lowest):
    from phl_risk.analysis import QuantileBinner

    reference = np.arange(101)
    options = dict(
        labels=["low", "mid", "upper", "high"], precision=3, include_lowest=include_lowest
    )
    quantile = QuantileBinner(4, **options).fit(reference)
    width = EqualWidthBinner(4, **options).fit(reference)
    fixed = FixedBinner(quantile.bin_edges_, **options)
    current = pd.Series(
        [-np.inf, -50, 0, 25, 25.001, 50, 75, 100, 200, np.inf, np.nan],
        index=[3] * 11,
        name="score",
    )
    original = current.copy(deep=True)
    for binner in [width, fixed]:
        pd.testing.assert_series_equal(binner.transform(current), quantile.transform(current))
        for key in ["bin_edges", "labels", "n_bins", "intervals", "include_lowest"]:
            assert binner.metadata()[key] == quantile.metadata()[key]
    pd.testing.assert_series_equal(current, original)


@pytest.mark.parametrize("kind", ["quantile", "width"])
def test_learned_binner_failed_refit_preserves_complete_state(kind):
    from phl_risk.analysis import QuantileBinner

    cls = QuantileBinner if kind == "quantile" else EqualWidthBinner
    binner = cls(2, labels=["low", "high"]).fit([0, 20, 50, 100])
    before = binner.metadata()
    for invalid in [[np.nan, np.inf], ["not a number"], [1, 1]]:
        with pytest.raises(TransformError):
            binner.fit(invalid)
        assert binner.metadata() == before
