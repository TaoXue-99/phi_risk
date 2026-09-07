"""Independent arithmetic checks for quantity, predicate and funnel contracts."""

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    AnalysisContext,
    BinDimension,
    Col,
    ComputePolicy,
    CountWhere,
    Cube,
    Funnel,
    QuantileBinner,
    Ratio,
    Stage,
    Sum,
    Transition,
)
from phl_risk.exceptions import EngineError, InvalidMetricError, MeasureError


def values(result):
    return result.data_.set_index("metric")["value"].to_dict()


def test_aggregate_counts_and_pooled_totals():
    data = pd.DataFrame({"dt": ["a", "b"], "浏览": [100, 10], "点击": [10, 8], "确认": [5, 4]})
    funnel = Funnel(["浏览", "点击", "确认"], rates="both")
    result = Cube(["dt"], funnel.measures()).compute(data, totals=True)
    total = values(result.total(over=["dt"]))
    assert total["浏览数量"] == 110
    assert total["浏览→点击转化率"] == pytest.approx(18 / 110)
    assert total["点击→确认转化率"] == 0.5
    assert total["浏览→确认转化率"] == pytest.approx(9 / 110)
    assert total["浏览→点击转化率"] != (0.1 + 0.8) / 2
    assert result.layout(totals=True, total_label="总计").loc["总计", "浏览数量"] == 110


def test_flags_and_reference_bins_are_independent_of_funnel():
    data = pd.DataFrame({"score": [0.0, 1.0, 2.0, 3.0], "a": [1, 1, 1, 1], "b": [0, 1, 1, 0]})
    cube = Cube([BinDimension("score", QuantileBinner(2))], Funnel(["a", "b"]).measures()).fit(data)
    edges = cube.dimensions_[0].transformer.bin_edges_.copy()
    current = data.iloc[[0, 0, 3]].copy()
    result = cube.compute(current, totals=True)
    np.testing.assert_array_equal(cube.dimensions_[0].transformer.bin_edges_, edges)
    assert values(result.total(over=["score_bin"]))["a数量"] == 3
    table = result.layout(bin_labels="interval", totals=True)
    assert table["a数量"].iloc[:2].tolist() == [2, 1]
    pd.testing.assert_frame_equal(
        Cube([BinDimension("score", QuantileBinner(2))], Funnel(["a", "b"]).measures())
        .fit_compute(data)
        .data_,
        cube.compute(data).data_,
    )


def test_explicit_stage_predicates_and_transitions():
    data = pd.DataFrame(
        {"status": ["view", "click", "done", None], "ok": [1, 1, 1, 1]}, index=[0, 0, 0, 0]
    )
    funnel = Funnel(
        [
            Stage("浏览", CountWhere(Col("status").notna())),
            Stage("点击", CountWhere(Col("status").isin(["click", "done"]) & (Col("ok") == 1))),
            Stage("确认", CountWhere(Col("status") == "done")),
        ],
        rates=[Transition("浏览", "确认", name="最终转化")],
    )
    actual = values(Cube([], funnel.measures()).compute(data))
    assert actual == {"浏览数量": 3, "点击数量": 2, "确认数量": 1, "最终转化": 1 / 3}


@pytest.mark.parametrize("mode,n", [("adjacent", 7), ("from_first", 7), ("both", 9)])
def test_dynamic_stage_counts(mode, n):
    assert len(Funnel(["a", "b", "c", "d"], rates=mode).measures()) == n
    assert len(Funnel(["a"], rates=mode).measures()) == 1
    assert len(Funnel(["a", "b"], rates=mode).measures()) == 3
    assert len(Funnel(["a", "b"], rates=[]).measures()) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stages": []},
        {"stages": "a"},
        {"stages": ["a", "a"]},
        {"stages": ["a"], "rates": "bad"},
        {"stages": ["a"], "aggregation": "count"},
        {"stages": ["a", "b"], "rates": [Transition("b", "a")]},
        {"stages": ["a", "b"], "rates": [Transition("a", "a")]},
        {"stages": ["a", "b"], "rates": [Transition("a", "c")]},
        {"stages": ["a", "b"], "rates": [Transition("a", "b"), Transition("a", "b")]},
        {"stages": ["a", "b"], "rates": [Transition("a", "b", "a数量")]},
    ],
)
def test_invalid_funnel_configuration(kwargs):
    with pytest.raises(MeasureError):
        Funnel(**kwargs)


def test_ratio_topological_evaluation_preserves_display_order():
    measures = [Ratio("r", "b", "nested"), Ratio("a", "b", "r"), Sum("x", "a"), Sum("y", "b")]
    cube = Cube([], measures)
    assert cube.plan().required_columns() == ("x", "y")
    result = cube.compute(pd.DataFrame({"x": [4, 8], "y": [1, 2]}))
    assert list(values(result)) == ["nested", "r", "a", "b"]
    assert values(result)["nested"] == pytest.approx(4 / 3)
    assert "a / b" in cube.explain().to_string()


@pytest.mark.parametrize(
    "measures",
    [
        [Ratio("absent", "absent")],
        [Ratio("b", "b", "a"), Ratio("a", "a", "b")],
        [Ratio("a", "a", "a")],
    ],
)
def test_invalid_references_fail_at_plan_time(measures):
    with pytest.raises(MeasureError):
        Cube([], measures).plan()


def test_sum_missing_empty_and_context_weight():
    data = pd.DataFrame({"x": pd.Series([2, None], dtype="Int64"), "w": [100, 1]})
    measures = [Sum("x", "unknown"), Sum("x", "zero", missing="zero")]
    result = values(Cube([], measures).compute(data, context=AnalysisContext(weight="w")))
    assert np.isnan(result["unknown"])
    assert result["zero"] == 2
    assert values(Cube([], measures).compute(data.iloc[:0])) == {"unknown": 0, "zero": 0}
    assert values(Cube([], measures).compute(data.iloc[1:]))["zero"] == 0


@pytest.mark.parametrize("column", [["1", "2"], [1 + 2j], [np.inf]])
def test_sum_rejects_non_numeric_or_infinite(column):
    with pytest.raises(EngineError):
        Cube([], [Sum("x")]).compute(pd.DataFrame({"x": column}))


def test_predicate_unknowns_boolean_logic_and_deduplication():
    data = pd.DataFrame({"x": [1.0, 0.0, np.nan]})
    condition = ~(Col("x") == 1)
    measures = [
        CountWhere(condition, "a"),
        CountWhere(~(Col("x") == 1), "b"),
        CountWhere(Col("x").isna(), "null"),
    ]
    cube = Cube([], measures)
    assert len(cube.plan().aggregates) == 2
    assert values(cube.compute(data)) == {"a": 1, "b": 1, "null": 1}
    assert values(Cube([], [Sum("x")], filters=[Col("x") > 0]).compute(data))["sum__x"] == 1
    with pytest.raises(TypeError):
        bool(condition)
    with pytest.raises(MeasureError):
        Col("x") == None  # noqa: E711


@pytest.mark.parametrize("policy", ["nan", "warn", "raise"])
def test_zero_denominator_policy(policy):
    cube = Cube([], Funnel(["a", "b"]).measures(), policy=ComputePolicy(on_invalid=policy))
    data = pd.DataFrame({"a": [0], "b": [0]})
    if policy == "raise":
        with pytest.raises(InvalidMetricError, match="zero denominator"):
            cube.compute(data)
    elif policy == "warn":
        with pytest.warns(RuntimeWarning, match="zero denominator"):
            cube.compute(data)
    else:
        assert np.isnan(values(cube.compute(data))["a→b转化率"])


def test_grouped_missing_and_empty_categories():
    data = pd.DataFrame(
        {
            "dt": pd.Categorical(["x", "x", "y"], categories=["x", "y", "z"]),
            "a": [1.0, np.nan, 2.0],
            "b": [0, 0, 1],
        }
    )
    result = Cube(["dt"], Funnel(["a", "b"]).measures()).compute(data, totals=True)
    table = result.layout()
    assert np.isnan(table.loc["x", "a数量"])
    assert table.loc["y", "a→b转化率"] == 0.5
    assert table.loc["z", "a数量"] == 0
    assert np.isnan(table.loc["z", "a→b转化率"])
    assert np.isnan(values(result.total(over=["dt"]))["a→b转化率"])


def test_two_dimensional_margins_and_shared_native_reduction(monkeypatch):
    frame = pd.DataFrame(
        {"dt": ["a", "a", "b"], "channel": ["x", "y", "x"], "a": [10, 20, 100], "b": [5, 2, 10]}
    )
    calls = []
    original = pd.DataFrame.groupby

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", spy)
    result = Cube(["dt", "channel"], Funnel(["a", "b"]).measures()).compute(frame, totals=True)
    assert len(calls) == 3  # Full grouping, dt margin, channel margin; global has no groupby.
    table = result.layout(rows=["metric", "dt"], columns=["channel"], totals=True)
    assert table.loc[("a→b转化率", "a"), "Total"] == pytest.approx(7 / 30)
    assert table.loc[("a→b转化率", "Total"), "x"] == pytest.approx(15 / 110)
    assert table.loc[("a→b转化率", "Total"), "Total"] == pytest.approx(17 / 130)


@pytest.mark.parametrize("dimensions", [[], ["dt"]])
@pytest.mark.parametrize("policy", ["nan", "raise"])
def test_numeric_overflow_respects_policy(dimensions, policy):
    data = pd.DataFrame({"dt": ["x", "x"], "a": [1e308, 1e308]})
    cube = Cube(dimensions, [Sum("a")], policy=ComputePolicy(on_invalid=policy))
    if policy == "raise":
        with pytest.raises(InvalidMetricError, match="non-finite aggregate"):
            cube.compute(data)
    else:
        assert np.isnan(cube.compute(data).data_["value"]).all()


def test_ratio_overflow_and_negative_denominator():
    data = pd.DataFrame({"a": [1e308], "b": [1e-308], "c": [-2.0]})
    measures = [
        Sum("a", "a"),
        Sum("b", "b"),
        Sum("c", "c"),
        Ratio("a", "b", "huge"),
        Ratio("c", "c", "one"),
    ]
    result = values(Cube([], measures).compute(data))
    assert np.isnan(result["huge"])
    assert result["one"] == 1
