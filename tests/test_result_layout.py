from itertools import permutations

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
    CubeResult,
    EventRate,
    MissingPolicy,
    QuantileBinner,
    Share,
    TableLayout,
)
from phl_risk.exceptions import LayoutError


def cross_cube():
    return Cube(
        [BinDimension("score_a", QuantileBinner(5)), BinDimension("score_b", QuantileBinner(5))],
        [Count(), Share(), EventRate("label", name="event_rate")],
    )


def test_cross_five_by_five_and_layout_orders():
    train = pd.DataFrame({"score_a": range(100), "score_b": range(100)})
    current = pd.DataFrame(
        {
            "score_a": np.repeat([5, 25, 45, 65, 85], 5),
            "score_b": np.tile([5, 25, 45, 65, 85], 5),
            "label": [0, 1, 0, 1, 0] * 5,
        }
    )
    cube = cross_cube().fit(train)
    result = cube.compute(current)
    assert result.shape == (5, 5, 3) and result.ndim == 3
    original = result.data_
    # Layout has no reference to Cube or Engine, and must work after either is gone.
    del cube
    table = result.layout(rows=["metric", "score_a_bin"], columns=["score_b_bin"])
    expected = [(m, f"B{i}") for m in ["count", "share", "event_rate"] for i in range(1, 6)]
    assert table.index.tolist() == expected
    assert table.index.names == ["metric", "score_a_bin"]
    assert table.columns.tolist() == ["B1", "B2", "B3", "B4", "B5"]
    assert table.index.levels[0].tolist() == ["count", "share", "event_rate"]
    np.testing.assert_array_equal(table.loc["count"], np.ones((5, 5)))
    np.testing.assert_allclose(table.loc["share"], 1 / 25)
    np.testing.assert_array_equal(table.loc["event_rate"], np.tile([0, 1, 0, 1, 0], (5, 1)))
    other = result.layout(rows=["score_a_bin"], columns=["metric", "score_b_bin"])
    assert other.columns.names == ["metric", "score_b_bin"]
    assert other.columns.tolist() == expected
    np.testing.assert_array_equal(other["count"], table.loc["count"])
    pd.testing.assert_frame_equal(original, result.data_)


def test_empty_bins_and_sparse_canonical():
    train = pd.DataFrame({"score_a": range(100), "score_b": range(100)})
    cube = cross_cube().fit(train)
    current = pd.DataFrame({"score_a": [-999, 999], "score_b": [-999, 999], "label": [0, 1]})
    result = cube.compute(current)
    assert result.shape == (5, 5, 3) and len(result.data_) == 6
    table = result.layout(["metric", "score_a_bin"], ["score_b_bin"])
    assert table.loc[("count", "B3"), "B3"] == 0
    assert table.loc[("share", "B3"), "B3"] == 0
    assert np.isnan(table.loc[("event_rate", "B3"), "B3"])
    assert table.loc[("share", "B1"), "B1"] == 0.5
    empty = cube.compute(current.iloc[:0]).layout(["metric", "score_a_bin"], ["score_b_bin"])
    assert (empty.loc["count"] == 0).all().all()
    assert empty.loc["share"].isna().all().all()


def test_stratified_layout_and_all_axis_permutations(fixture_analysis_df):
    result = Cube(["dt", "user_type"], [AUC("score", "label"), KS("score", "label")]).compute(
        fixture_analysis_df
    )
    standard = result.layout(["dt", "user_type"], ["metric"])
    assert standard.index.names == ["dt", "user_type"]
    other = result.layout(["dt"], ["user_type", "metric"])
    assert other.columns.names == ["user_type", "metric"]
    assert other.loc["2026-01", ("new", "auc__score")] == 0.75
    names = [a.name for a in result.axes]
    for order in permutations(names):
        for split in range(4):
            table = TableLayout(order[:split], order[split:]).render(result)
            assert table.size == np.prod(result.shape)
            np.testing.assert_allclose(
                np.sort(table.to_numpy().ravel()), np.sort(result.data_.value.to_numpy())
            )


def test_categorical_order_and_missing_coordinates():
    df = pd.DataFrame(
        {
            "g": pd.Categorical(["a", None], categories=["c", "b", "a"], ordered=True),
            "h": ["z", None],
        }
    )
    result = Cube(
        ["g", "h"], [Count(), Share()], policy=ComputePolicy(MissingPolicy(dimension="keep"))
    ).compute(df)
    assert result.axes[0].values == ("c", "b", "a", None)
    assert result.shape == (4, 2, 2)
    table = result.layout(["metric", "g"], ["h"])
    assert table.loc["count"].to_numpy().sum() == 2
    assert table.loc["share"].to_numpy().sum() == 1


def test_defensive_result_and_layout_validation(fixture_analysis_df):
    result = Cube(["dt"]).compute(fixture_analysis_df)
    exposed = result.data_
    exposed.loc[:, "value"] = -1
    assert (result.data_.value >= 0).all()
    metadata = result.metadata_
    metadata["dimensions"][0]["name"] = "corrupted"
    assert result.metadata_["dimensions"][0]["name"] == "dt"
    for rows, columns in [(["dt"], []), (["dt"], ["dt", "metric"]), (["absent"], ["metric"])]:
        with pytest.raises(LayoutError):
            result.layout(rows, columns)
    with pytest.raises(LayoutError, match="max_cells"):
        result.layout(max_cells=1)
    with pytest.raises(LayoutError, match="Duplicate canonical"):
        CubeResult(pd.concat([result.data_, result.data_]), result.axes, result.measures_)
