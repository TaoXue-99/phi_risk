import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    AUC,
    KS,
    BinDimension,
    ComputePolicy,
    Count,
    Cube,
    EventRate,
    MissingPolicy,
    PandasEngine,
    QuantileBinner,
    Share,
)
from phl_risk.analysis.measures._base import validate_field
from phl_risk.exceptions import (
    AnalysisError,
    CubeError,
    InvalidMetricError,
    LayoutError,
    MeasureError,
    NotFittedError,
    PhlRiskError,
    TransformError,
)
from phl_risk.metrics import auc_score, event_rate, ks_score


def test_exception_hierarchy_and_validate_field():
    assert issubclass(NotFittedError, TransformError)
    assert issubclass(TransformError, AnalysisError)
    assert issubclass(InvalidMetricError, (PhlRiskError, ValueError))
    assert not issubclass(InvalidMetricError, AnalysisError)
    assert validate_field("label", "target") == "label"
    assert validate_field(None, "target", required=False) is None
    with pytest.raises(MeasureError):
        validate_field("", "target")


def test_pooled_auc_ks_not_cell_mean_and_unstack(monkeypatch):
    frame = pd.DataFrame(
        {
            "dt": ["d1", "d1", "d2", "d2"] * 2,
            "dataset": ["train"] * 4 + ["oot"] * 4,
            "y": [0, 1, 0, 1] * 2,
            "s": [0.8, 0.9, 0.1, 0.2, 0.9, 0.8, 0.2, 0.1],
        }
    )
    cube = Cube(["dt", "dataset"], [AUC("s", "y"), KS("s", "y")])
    ordinary = cube.compute(frame)
    result = cube.compute(frame, totals=True)
    pd.testing.assert_frame_equal(ordinary.data_, result.data_)
    assert ordinary.shape == result.shape
    assert result.metadata_["total_grouping_sets"] == 3
    before = result.data_

    def reject(*args, **kwargs):
        raise AssertionError("layout must not execute")

    monkeypatch.setattr(PandasEngine, "execute", reject)
    table = result.layout(["dt"], ["metric", "dataset"], totals=["dt"])
    via_unstack = result.layout(totals=["dt"]).unstack("dataset")
    pd.testing.assert_frame_equal(table, via_unstack)
    assert table.index[-1] == "Total"
    assert table.loc["Total", ("auc__s", "train")] == 0.75
    assert table.loc["Total", ("ks__s", "train")] == 0.5
    for dataset, group in frame.groupby("dataset"):
        assert table.loc["Total", ("auc__s", dataset)] == auc_score(group.y, group.s)
        assert table.loc["Total", ("ks__s", dataset)] == ks_score(group.y, group.s)
    pd.testing.assert_frame_equal(before, result.data_)
    assert result.total(over=["dt"]).axes[0].name == "dataset"


@pytest.mark.parametrize("weighted", [False, True])
def test_cross_margins_pooled_event_rate_and_intervals(weighted):
    train = pd.DataFrame({"a": range(11), "b": range(11)})
    current = pd.DataFrame(
        {
            "a": [0, 1, 2, 7, 9, 10],
            "b": [1, 9, 10, 2, 8, 10],
            "y": [1, 0, None, 0, 1, 1],
            "w": [3, 1, 99, 2, 4, 2],
        }
    )
    weight = "w" if weighted else None
    cube = Cube(
        [BinDimension("a", QuantileBinner(2)), BinDimension("b", QuantileBinner(2))],
        [Count(), Share(), EventRate("y", weight=weight, name="bad_rate")],
    ).fit(train)
    result = cube.compute(current, totals=True)
    table = result.layout(["metric", "a_bin"], ["b_bin"], totals=True)
    assert table.index.tolist() == [
        (m, b) for m in ["count", "share", "bad_rate"] for b in ["B1", "B2", "Total"]
    ]
    assert table.columns.tolist() == ["B1", "B2", "Total"]
    assert table.loc[("count", "Total"), "Total"] == 6
    assert table.loc[("share", "Total"), "Total"] == 1
    assert table.loc[("bad_rate", "Total"), "Total"] == pytest.approx(
        event_rate(current.y, sample_weight=current.w if weighted else None)
    )
    for name, axis in [("a", "row"), ("b", "column")]:
        bins = cube.dimensions_[0 if name == "a" else 1].transform(current)
        for label in ["B1", "B2"]:
            group = current.loc[bins == label]
            expected = event_rate(group.y, sample_weight=group.w if weighted else None)
            actual = (
                table.loc[("bad_rate", label), "Total"]
                if axis == "row"
                else table.loc[("bad_rate", "Total"), label]
            )
            assert actual == pytest.approx(expected)
    intervals = result.layout(
        ["metric", "a_bin"], ["b_bin"], totals=True, bin_labels="interval", total_label="总计"
    )
    assert intervals.columns.tolist() == ["[-inf, 5.000000]", "(5.000000, inf]", "总计"]
    np.testing.assert_allclose(intervals.to_numpy(), table.to_numpy(), equal_nan=True)
    assert result.axes[0].values == ("B1", "B2")


def test_population_filters_missing_and_no_repeat_transforms(monkeypatch):
    frame = pd.DataFrame({"a": [1, 2, None, 3], "b": [1, None, 2, 3], "y": [1, 0, 0, 0]})
    calls = []

    def predicate(data):
        calls.append(1)
        return data.a != 3

    cube = Cube(["a", "b"], [Count(), Share(), EventRate("y")], filters=[predicate])
    result = cube.compute(frame, totals=True)
    assert len(calls) == 1
    total = result.total(over=["a", "b"]).layout()
    assert total.loc[0, "count"] == 1  # Other rows stay excluded even after collapsing axes.
    assert total.loc[0, "share"] == 1
    assert total.loc[0, "event_rate__y__1"] == 1

    original = QuantileBinner.transform
    transforms = []

    def record(self, X):
        transforms.append(1)
        return original(self, X)

    monkeypatch.setattr(QuantileBinner, "transform", record)
    Cube([BinDimension("a", QuantileBinner(2)), BinDimension("b", QuantileBinner(2))]).fit_compute(
        frame, totals=True
    )
    assert len(transforms) == 2


def test_empty_bins_and_empty_population_totals():
    train = pd.DataFrame({"x": range(10), "z": range(10)})
    cube = Cube(
        [BinDimension("x", QuantileBinner(2)), BinDimension("z", QuantileBinner(2))],
        [Count(), Share(), EventRate("y")],
    ).fit(train)
    current = pd.DataFrame({"x": [0], "z": [0], "y": [1]})
    table = cube.compute(current, totals=True).layout(["metric", "x_bin"], ["z_bin"], totals=True)
    assert table.loc[("count", "B2"), "Total"] == 0
    assert np.isnan(table.loc[("event_rate__y__1", "B2"), "Total"])
    empty = cube.compute(current.iloc[:0], totals=True).layout(
        ["metric", "x_bin"], ["z_bin"], totals=True
    )
    assert empty.loc["count"].to_numpy().sum() == 0
    assert empty.loc["share"].isna().all().all()


def test_missing_kept_and_total_validation():
    frame = pd.DataFrame({"g": [None, "a"]})
    cube = Cube(["g"], policy=ComputePolicy(MissingPolicy(dimension="keep")))
    plain = cube.compute(frame)
    with pytest.raises(LayoutError, match="totals=True"):
        plain.layout(totals=True)
    result = cube.compute(frame, totals=True)
    assert result.layout(totals=True).iloc[-1, 0] == 2
    for axes in (["metric"], ["g", "g"], ["absent"]):
        with pytest.raises(LayoutError):
            result.layout(totals=axes)
    with pytest.raises(LayoutError, match="collides"):
        result.layout(totals=True, total_label="a")
    with pytest.raises(LayoutError, match="max_cells"):
        result.layout(totals=True, max_cells=2)
    with pytest.raises(LayoutError):
        result.layout(bin_labels="unknown")
    assert Cube().compute(frame, totals=True).layout(totals=True).iloc[0, 0] == 2


def test_explain_table_and_plain_text():
    cube = Cube(["dt", "dataset"], [AUC("s", "y"), KS("s", "y")])
    table = cube.explain(totals=True)
    assert table.columns.tolist() == ["Section", "Item", "Description"]
    assert "AUC: target=y, score=s" in table.Description.tolist()
    assert "Recompute pooled samples" in table.Description.tolist()
    text = cube.explain(format="text")
    assert "Cube Analysis Plan" in text and "auc__s" in text
    with pytest.raises(CubeError):
        cube.explain(format="unknown")
    bin_cube = Cube([BinDimension("s", QuantileBinner(2))]).fit(pd.DataFrame({"s": [0, 1, 2]}))
    assert "(-inf, 1.0, inf)" in bin_cube.explain().Description.tolist()
