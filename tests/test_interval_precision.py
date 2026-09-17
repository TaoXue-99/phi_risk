import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import BinDimension, Count, Cube, QuantileBinner


def test_default_six_decimal_places_and_layout_share_labels():
    # Two different values yield a single finite median boundary.
    frame = pd.DataFrame({"x": [122.123456789, 124.123456789]})
    cube = Cube([BinDimension("x", QuantileBinner(2))], [Count()]).fit(frame)
    intervals = cube.dimensions_[0].transformer.metadata()["intervals"]
    assert intervals == ("[-inf, 123.123457]", "(123.123457, inf]")
    result = cube.compute(frame, totals=True)
    assert result.layout(bin_labels="interval", totals=True).index.tolist() == [*intervals, "Total"]


@pytest.mark.parametrize("precision", [0, 2, 6])
def test_display_only_preserves_actual_edges_and_assignments(precision):
    source = pd.Series([0.123456789, 1.234567891, 2.345678912, 3.456789123])
    normal = QuantileBinner(2).fit(source)
    custom = QuantileBinner(2, precision=precision).fit(source)
    edge = normal.bin_edges_[1]
    probe = pd.Series([np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)])
    np.testing.assert_array_equal(normal.bin_edges_, custom.bin_edges_)
    pd.testing.assert_series_equal(normal.transform(probe), custom.transform(probe))
    assert custom.metadata()["intervals"][0] == f"[-inf, {edge:.{precision}f}]"


def test_close_edges_gain_precision_and_share_endpoints():
    source = pd.Series([0.12345671, 0.12345672, 0.12345673, 0.12345674])
    binner = QuantileBinner(4, precision=2).fit(source)
    labels = binner.metadata()["intervals"]
    endpoints = [label[1:-1].split(", ") for label in labels]
    for left, right in endpoints:
        assert float(left) < float(right)
    assert all(a[1] == b[0] for a, b in zip(endpoints, endpoints[1:]))
    assert labels[0].startswith("[-inf, 0.1234567")


def test_infinity_and_constant_interval():
    assert QuantileBinner(5).fit([1, 1]).metadata()["intervals"] == ("[-inf, inf]",)
    assert (
        QuantileBinner(2, include_lowest=False).fit([0, 1]).metadata()["intervals"][0]
        == "(-inf, 0.500000]"
    )


@pytest.mark.parametrize("scale", [1e-8, np.nextafter(0.0, 1.0)])
def test_tiny_signed_boundaries_remain_distinct(scale):
    source = pd.Series(np.arange(-4, 5, dtype=float) * scale)
    binner = QuantileBinner(4, precision=0).fit(source)
    intervals = binner.metadata()["intervals"]
    for label in intervals:
        left, right = label[1:-1].split(", ")
        assert float(left) < float(right)
    cube = Cube([BinDimension("x", binner)], [Count()]).fit(source.to_frame("x"))
    result = cube.compute(source.to_frame("x"))
    assert result.layout(bin_labels="interval").index.tolist() == list(intervals)
