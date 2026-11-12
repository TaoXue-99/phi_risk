"""Opt-in within-stratum bins: reference lifecycle, pooling and interval display."""

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    BinDimension,
    ColumnDimension,
    ComputePolicy,
    Count,
    Cube,
    EventRate,
    MissingPolicy,
    QuantileBinner,
)
from phl_risk.exceptions import DimensionError, LayoutError, NotFittedError, TransformError


def sample():
    return pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": list(range(10)) + list(range(100, 110)),
            "label": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1] * 2,
        }
    )


def cube(*, policy=None, bins=5):
    return Cube(
        ["group", BinDimension("score", QuantileBinner(bins), fit_by=["group"])],
        [Count(), EventRate("label", name="rate")],
        policy=policy,
    )


def test_group_bins_equal_independent_cubes_and_global_is_unchanged():
    frame = sample()
    model = cube()
    result = model.fit_compute(frame)
    for key, group in frame.groupby("group"):
        expected = Cube(
            [BinDimension("score", QuantileBinner(5))], [Count(), EventRate("label", name="rate")]
        ).fit_compute(group)
        actual = result.data_.loc[lambda d: d.group.eq(key)].drop(columns="group")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.data_)
    metadata = model.dimensions_[1].metadata()
    assert metadata["fit_by"] == ("group",)
    assert metadata["groups"][0]["key"] == ("A",)
    assert metadata["groups"][0]["transform"]["bin_edges"][1] == pytest.approx(1.8)
    assert metadata["groups"][1]["transform"]["bin_edges"][1] == pytest.approx(101.8)
    global_result = Cube(["group", BinDimension("score", QuantileBinner(5))]).fit_compute(frame)
    assert "fit_by" not in global_result.metadata_["dimensions"][1]


def test_reference_reuse_unknown_group_and_atomic_refit():
    model = cube().fit(sample())
    before = model.dimensions_[1].metadata()
    current = pd.DataFrame({"group": ["A", "B"], "score": [105, 2], "label": [1, 0]})
    assert model.compute(current).data_.score_bin.tolist() == ["B5", "B1"] * 2
    with pytest.raises(DimensionError, match="[Uu]nseen.*C"):
        model.compute(current.assign(group="C"))
    broken = sample()
    broken.loc[broken.group.eq("B"), "score"] = np.nan
    with pytest.raises(TransformError, match="B"):
        model.fit(broken)
    assert model.dimensions_[1].metadata() == before
    assert len(model.compute(current).data_) == 4
    with pytest.raises(NotFittedError):
        cube().compute(sample())


def test_duplicate_indices_multikey_and_fit_field():
    frame = sample().assign(channel=["x", "y"] * 10, learn=lambda d: d.score * 2)
    frame.index = [3] * len(frame)
    dim = BinDimension("score", QuantileBinner(2), fit_field="learn", fit_by=["group", "channel"])
    model = Cube(["channel", dim, "group"], [Count()]).fit(frame)
    result = model.compute(frame)
    assert result.data_.value.sum() == len(frame)
    for record in model.dimensions_[1].metadata()["groups"]:
        g, c = record["key"]
        source = frame.loc[frame.group.eq(g) & frame.channel.eq(c)]
        assert record["transform"]["bin_edges"][1] == source.learn.median()
    transformed = model.dimensions_[1].transform(frame)
    assert transformed.index.equals(frame.index)


def test_missing_group_policy():
    frame = sample()
    frame.loc[:4, "group"] = None
    frame.loc[:4, "score"] = np.nan
    dropped = cube().fit_compute(frame)
    assert dropped.metadata_["analyzed_rows"] == 15
    frame.loc[:4, "score"] = range(5)
    kept = cube(policy=ComputePolicy(missing=MissingPolicy(dimension="keep")))
    result = kept.fit_compute(frame, totals=True)
    assert result.metadata_["analyzed_rows"] == 20
    assert any(g["key"] == (None,) for g in kept.dimensions_[1].metadata()["groups"])
    table = result.layout(rows=["group", "score_bin"], totals=True, bin_labels="interval")
    assert table.shape[1] == 2
    assert result.total(over=["score_bin"]).data_.group.isna().sum() == 2


@pytest.mark.parametrize("fit_by", [[], "group", ["group", "group"], [""], [1]])
def test_invalid_fit_by(fit_by):
    with pytest.raises(DimensionError, match="fit_by"):
        BinDimension("score", QuantileBinner(5), fit_by=fit_by)


def test_grouping_fields_must_be_unrenamed_raw_dimensions():
    dim = BinDimension("score", QuantileBinner(), fit_by=["group"])
    for dimensions in ([dim], [ColumnDimension("group", name="renamed"), dim]):
        with pytest.raises(DimensionError, match="fit_by"):
            Cube(dimensions)


def test_group_only_totals_and_invalid_explicit_requests():
    result = cube().fit_compute(sample(), totals=True)
    assert result.metadata_["total_grouping_sets"] == 1
    total = result.total(over=["score_bin"]).layout()
    assert total.loc["A", "count"] == 10
    assert total.loc["A", "rate"] == pytest.approx(0.6)
    for over in (["group"], ["group", "score_bin"]):
        with pytest.raises(LayoutError, match="fit_by|stratif"):
            result.total(over=over)
        with pytest.raises(LayoutError, match="fit_by|stratif"):
            result.layout(totals=over)
    table = result.layout(
        rows=["group", "metric", "score_bin"],
        totals=True,
        total_label="总计",
        bin_labels="interval",
    )
    assert "总计" not in table.index.get_level_values("group")
    assert table.loc[("A", "count", "总计")].iloc[0] == 10
    assert table.loc[("A", "count", "[-inf, 1.800000]")].iloc[0] == 2
    assert table.loc[("B", "count", "[-inf, 101.800000]")].iloc[0] == 2


def test_contextual_intervals_on_columns_and_split_axes():
    result = cube().fit_compute(sample(), totals=True)
    columns = result.layout(
        rows=["metric"], columns=["group", "score_bin"], totals=True, bin_labels="interval"
    )
    assert columns.loc["count", ("B", "[-inf, 101.800000]")] == 2
    split = result.layout(
        rows=["score_bin"], columns=["group", "metric"], totals=True, bin_labels="interval"
    )
    assert split.loc["[-inf, 1.800000]", ("A", "count")] == 2
    assert pd.isna(split.loc["[-inf, 1.800000]", ("B", "count")])
    with pytest.raises(LayoutError, match="cells"):
        result.layout(max_cells=2, bin_labels="interval")


def test_varying_actual_bins_custom_labels_empty_and_defensive_metadata():
    frame = sample()
    frame.loc[frame.group.eq("B"), "score"] = 100
    model = cube().fit(frame)
    result = model.compute(frame, totals=True)
    table = result.layout(rows=["group", "score_bin"], bin_labels="interval", totals=True)
    assert len(table.loc["B"]) == 2  # one actual bin and the group total
    assert table.loc[("B", "[-inf, inf]"), "count"] == 10
    assert model.compute(frame.iloc[:0]).layout(bin_labels="interval").empty
    meta = model.dimensions_[1].metadata()
    meta["groups"][0]["transform"]["bin_edges"] = ()
    assert model.dimensions_[1].metadata()["groups"][0]["transform"]["bin_edges"]
    labeled = Cube(
        [
            "group",
            BinDimension("score", QuantileBinner(2, labels=["low", "high"]), fit_by=["group"]),
        ]
    )
    assert set(labeled.fit_compute(sample()).data_.score_bin) == {"low", "high"}


def test_two_independent_stratified_dimensions_and_mixed_global_bins():
    frame = sample().assign(channel=["x", "y"] * 10)
    model = Cube(
        [
            "group",
            "channel",
            BinDimension("score", QuantileBinner(2), fit_by=["group"], name="group_bin"),
            BinDimension("score", QuantileBinner(2), fit_by=["channel"], name="channel_bin"),
            BinDimension("score", QuantileBinner(2), name="global_bin"),
        ]
    )
    result = model.fit_compute(frame, totals=True)
    assert result.metadata_["total_grouping_sets"] == 7
    assert result.total(over=["group_bin", "channel_bin", "global_bin"]).data_.value.sum() == 20
    table = result.layout(bin_labels="interval", totals=True)
    assert table["count"].sum() > 0


@pytest.mark.parametrize("duplicate_index", [False, True])
@pytest.mark.parametrize("drop_missing", [False, True])
def test_filtered_supervised_transformer_receives_aligned_groups(duplicate_index, drop_missing):
    from phl_risk.analysis import BaseTransformer

    class SupervisedBins(BaseTransformer):
        def fit(self, X, y=None):
            np.testing.assert_array_equal(X.to_numpy(), np.asarray(y))
            self.binner = QuantileBinner(2).fit(X)
            return self

        @property
        def is_fitted(self):
            return hasattr(self, "binner")

        def transform(self, X):
            return self.binner.transform(X)

        def metadata(self):
            return self.binner.metadata() if self.is_fitted else {}

    frame = sample()
    if drop_missing:
        frame.loc[1, "group"] = None
    if duplicate_index:
        frame.index = [7] * len(frame)
    model = Cube(
        ["group", BinDimension("score", SupervisedBins(), fit_by=["group"])],
        filters=[lambda d: d.score.ne(0)],
    )
    assert model.fit_compute(frame, y=frame.score).data_.value.sum() == 19 - int(drop_missing)


def test_direct_plan_rejects_missing_stratification_axis():
    from phl_risk.analysis._plan import CubePlan

    with pytest.raises(DimensionError, match="fit_by"):
        CubePlan((BinDimension("score", QuantileBinner(), fit_by=["group"]),), ())


def test_interval_union_checks_final_cell_count():
    result = cube().fit_compute(sample())
    # Code grid: 5 * (2 * 2) = 20; interval union: 10 * (2 * 2) = 40.
    with pytest.raises(LayoutError, match="40 cells"):
        result.layout(
            rows=["score_bin"], columns=["group", "metric"], bin_labels="interval", max_cells=30
        )


def test_empty_layout_preserves_metric_columns():
    model = cube().fit(sample())
    table = model.compute(sample().iloc[:0]).layout(bin_labels="interval")
    assert table.empty
    assert table.columns.tolist() == ["count", "rate"]


def test_contextual_layout_respects_requested_axis_order():
    result = cube().fit_compute(sample(), totals=True)
    table = result.layout(rows=["group", "metric", "score_bin"], totals=True, bin_labels="interval")
    assert table.loc["A"].index.get_level_values("metric").tolist() == ["count"] * 6 + ["rate"] * 6
    columns = result.layout(
        rows=[], columns=["metric", "group", "score_bin"], bin_labels="interval"
    )
    assert columns.columns.get_level_values("metric").tolist() == ["count"] * 10 + ["rate"] * 10


def test_global_fit_retains_generic_backend_data_contract():
    from phl_risk.analysis import BaseCubeEngine

    class PassthroughEngine(BaseCubeEngine):
        def execute(self, plan, data, context=None):
            raise NotImplementedError

    assert Cube([], engine=PassthroughEngine()).fit(object()).is_fitted
