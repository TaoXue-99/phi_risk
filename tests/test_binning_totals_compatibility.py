"""Ordinary grouping must retain legacy totals unless fit_by is explicitly set."""

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import BinDimension, Count, Cube, EventRate, QuantileBinner
from phl_risk.exceptions import LayoutError


def frame():
    return pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 15,
            "score": np.arange(20, dtype=float),
            "target": [1] * 5 + [0] * 12 + [1] * 3,
        }
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_global_bins_keep_all_three_legacy_total_grains(explicit_none):
    data = frame()
    options = {"fit_by": None} if explicit_none else {}
    result = Cube(
        ["group", BinDimension("score", QuantileBinner(2), **options)],
        [Count(), EventRate("target", name="rate")],
    ).fit_compute(data, totals=True)
    assert "stratification_fields" not in result.metadata_
    assert result.metadata_["total_grouping_sets"] == 3
    group_totals = result.total(over=["score_bin"]).layout()
    assert group_totals.loc["A", "rate"] == 1
    assert group_totals.loc["B", "rate"] == pytest.approx(0.2)
    bin_totals = result.total(over=["group"]).layout()
    assert bin_totals.loc["B1", "count"] == 10
    assert bin_totals.loc["B2", "count"] == 10
    grand_total = result.total(over=["group", "score_bin"]).layout()
    assert grand_total.loc[0, "count"] == 20
    assert grand_total.loc[0, "rate"] == pytest.approx(0.4)
    table = result.layout(rows=["group", "score_bin"], totals=True, total_label="总计")
    assert table.loc[("总计", "总计"), "rate"] == pytest.approx(0.4)
    assert table.loc[("总计", "B1"), "count"] == 10
    assert table.loc[("A", "总计"), "count"] == 5
    # Explicit selective totals retain their original semantics too.
    selected = result.layout(totals=["group"], total_label="总计", bin_labels="interval")
    assert "总计" in selected.index.get_level_values("group")
    assert "总计" not in selected.index.get_level_values("score_bin")


def test_opt_in_changes_only_the_stratified_cube_not_later_global_cubes():
    data = frame()
    transformer = QuantileBinner(2)
    stratified = Cube(
        ["group", BinDimension("score", transformer, fit_by=["group"])],
        [Count(), EventRate("target", name="rate")],
    ).fit_compute(data, totals=True)
    assert stratified.metadata_["total_grouping_sets"] == 1
    group_totals = stratified.total(over=["score_bin"]).layout()
    assert group_totals.loc["A", "rate"] == 1
    assert group_totals.loc["B", "rate"] == pytest.approx(0.2)
    with pytest.raises(LayoutError, match="fit_by"):
        stratified.total(over=["group", "score_bin"])
    table = stratified.layout(totals=True, total_label="总计")
    assert "总计" not in table.index.get_level_values("group")
    assert "总计" in table.index.get_level_values("score_bin")
    # Even reusing the constructor transformer cannot leak fitted group state.
    legacy = Cube(["group", BinDimension("score", transformer)]).fit_compute(data, totals=True)
    assert legacy.metadata_["total_grouping_sets"] == 3
    assert legacy.total(over=["group", "score_bin"]).layout().loc[0, "count"] == 20
    assert not transformer.is_fitted


def test_plain_grouping_and_no_dimensions_keep_global_total():
    data = frame()
    grouped = Cube(["group"], [Count(), EventRate("target", name="rate")]).compute(
        data,
        totals=True,
    )
    assert grouped.layout(totals=True, total_label="总计").loc["总计", "rate"] == pytest.approx(0.4)
    overall = Cube([], [Count(), EventRate("target", name="rate")]).compute(data, totals=True)
    assert overall.layout(totals=True).loc[0, "count"] == 20
    assert overall.layout(totals=True).loc[0, "rate"] == pytest.approx(0.4)
