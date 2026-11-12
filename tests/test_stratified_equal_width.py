"""Equal-width strata learn each reference range, independently of quantiles."""

import logging

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    BinDimension,
    ComputePolicy,
    Count,
    Cube,
    EqualWidthBinner,
    EventRate,
    MissingPolicy,
)
from phl_risk.exceptions import DimensionError, LayoutError, TransformError


def sample():
    skewed = np.array([0, 1, 2, 3, 4, 5, 10, 20, 40, 100], dtype=float)
    return pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": np.r_[skewed, 100 + 2 * skewed],
            "label": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1] * 2,
        }
    )


def model(*, policy=None):
    return Cube(
        ["group", BinDimension("score", EqualWidthBinner(5), fit_by=["group"])],
        [Count(), EventRate("label", name="rate")],
        policy=policy,
    )


def test_equal_width_per_group_ranges_counts_and_totals(caplog):
    fitted = model()
    with caplog.at_level(logging.INFO, logger="phl_risk.analysis"):
        result = fitted.fit_compute(sample(), totals=True)
    assert not any(
        getattr(r, "event", None) == "analysis.binning.global_scope" for r in caplog.records
    )
    groups = fitted.dimensions_[1].metadata()["groups"]
    np.testing.assert_allclose(
        groups[0]["transform"]["bin_edges"], [-np.inf, 20, 40, 60, 80, np.inf]
    )
    np.testing.assert_allclose(
        groups[1]["transform"]["bin_edges"], [-np.inf, 140, 180, 220, 260, np.inf]
    )
    for key, frame in sample().groupby("group", sort=False):
        independent = Cube(
            [BinDimension("score", EqualWidthBinner(5))], [Count(), EventRate("label", name="rate")]
        ).fit_compute(frame)
        actual = result.data_.loc[lambda d: d.group.eq(key)].drop(columns="group")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), independent.data_)
    table = result.layout(totals=True)
    assert table.loc["A", "count"].tolist() == [8, 1, 0, 0, 1, 10]
    assert table.loc[("B", "Total"), "rate"] == pytest.approx(0.6)
    assert "Total" not in table.index.get_level_values("group")
    intervals = result.layout(totals=True, bin_labels="interval")
    assert intervals.loc[("B", "[-inf, 140.000000]"), "count"] == 8
    with pytest.raises(LayoutError, match="fit_by"):
        result.total(over=["group", "score_bin"])
    # No fit_by still means one global reference range and legacy full totals.
    global_model = Cube(["group", BinDimension("score", EqualWidthBinner(5))])
    global_result = global_model.fit_compute(sample(), totals=True)
    np.testing.assert_allclose(
        global_model.dimensions_[1].transformer.bin_edges_, [-np.inf, 60, 120, 180, 240, np.inf]
    )
    assert global_result.total(over=["group", "score_bin"]).layout().loc[0, "count"] == 20


def test_reference_boundaries_reused_unknown_group_and_atomic_refit():
    fitted = model().fit(sample())
    before = fitted.dimensions_[1].metadata()
    current = pd.DataFrame({"group": ["A", "B"], "score": [1000, -1000], "label": [1, 0]})
    assert fitted.compute(current).data_.score_bin.tolist() == ["B5", "B1"] * 2
    assert fitted.dimensions_[1].metadata() == before
    with pytest.raises(DimensionError, match="Unseen.*C"):
        fitted.compute(current.assign(group="C"))
    broken = sample()
    broken.loc[broken.group.eq("B"), "score"] = np.nan
    with pytest.raises(TransformError, match="B"):
        fitted.fit(broken)
    assert fitted.dimensions_[1].metadata() == before


def test_multifield_fit_field_and_duplicate_indices():
    frame = sample().assign(channel=["x", "y"] * 10, learn=lambda d: d.score * 2)
    frame.index = [1] * len(frame)
    fitted = Cube(
        [
            "group",
            "channel",
            BinDimension(
                "score",
                EqualWidthBinner(2),
                fit_by=["group", "channel"],
                fit_field="learn",
            ),
        ]
    ).fit(frame)
    for group in fitted.dimensions_[2].metadata()["groups"]:
        g, c = group["key"]
        source = frame.loc[frame.group.eq(g) & frame.channel.eq(c), "learn"]
        assert group["transform"]["bin_edges"][1] == pytest.approx(
            (source.min() + source.max()) / 2
        )
    result = fitted.compute(frame, totals=True)
    assert result.metadata_["total_grouping_sets"] == 1
    assert result.total(over=["score_bin"]).data_.value.sum() == 20
    assert fitted.dimensions_[2].transform(frame).index.equals(frame.index)


@pytest.mark.parametrize("missing", ["drop", "keep"])
def test_constant_stratum_missing_policy_and_actual_bin_counts(missing):
    frame = sample()
    frame.loc[:2, "group"] = None
    frame.loc[frame.group.eq("B"), "score"] = 100
    fitted = model(policy=ComputePolicy(missing=MissingPolicy(dimension=missing)))
    result = fitted.fit_compute(frame, totals=True)
    assert result.metadata_["analyzed_rows"] == (17 if missing == "drop" else 20)
    table = result.layout(totals=True, bin_labels="interval")
    assert len(table.loc["B"]) == 2  # one actual interval and group total
    assert table.loc[("B", "[-inf, inf]"), "count"] == 10
