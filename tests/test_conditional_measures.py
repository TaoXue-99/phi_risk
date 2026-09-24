import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import Col, Count, Cube, Sum
from phl_risk.exceptions import EngineError, MeasureError


def test_group_conditions_totals_and_shared_aggregates():
    data = pd.DataFrame(
        {
            "g": ["x", "x", "x", "y"],
            "cat": ["A", "B", None, "C"],
            "age": [20, 30, 40, 10],
            "amount": [20.0, 30.0, np.nan, 5.0],
        }
    )
    condition = Col("cat").isin(["A", "B"]) & (Col("age") >= 18)
    result = Cube(
        ["g"],
        [
            Count(),
            Count(where=condition, name="selected"),
            Sum("amount", where=condition, name="amount"),
            Count(where=condition, name="selected_again"),
        ],
    ).compute(data, totals=True)
    table = result.layout()
    assert table.loc["x"].tolist() == [3, 2, 50, 2]
    assert table.loc["y"].tolist() == [1, 0, 0, 0]
    assert result.total(over=["g"]).layout().iloc[0].tolist() == [4, 2, 50, 2]
    assert result.diagnostics().empty
    assert "condition=" in Cube(["g"], [Sum("amount", where=condition)]).explain(format="text")


def test_selected_missing_and_unknown_predicate():
    data = pd.DataFrame({"cat": ["A", "B", None], "v": [np.nan, 3.0, 4.0]})
    result = Cube(
        measures=[
            Sum("v", where=Col("cat") == "A"),
            Count(where=~(Col("cat") == "A"), name="not_a"),
            Count(where=Col("cat").isna(), name="missing"),
        ]
    ).compute(data)
    assert np.isnan(result.layout().iloc[0, 0])
    assert result.layout().iloc[0, 1:].tolist() == [1, 1]
    assert result.diagnostics().loc["sum__v", "affected_rows"] == 1
    assert (
        Cube(measures=[Sum("v", where=Col("cat") == "A", missing="zero")])
        .compute(data)
        .layout()
        .iloc[0, 0]
        == 0
    )


@pytest.mark.parametrize("cls,args", [(Count, ()), (Sum, ("v",))])
def test_where_validation_and_dependencies(cls, args):
    with pytest.raises(MeasureError, match="where"):
        cls(*args, where="v > 10")
    with pytest.raises(EngineError, match="absent"):
        Cube(measures=[cls(*args, where=Col("absent") > 10)]).compute(pd.DataFrame({"v": [1]}))


def test_numeric_threshold_and_excluded_infinity():
    data = pd.DataFrame({"v": [5.0, 20.0, 30.0, np.nan], "keep": [True, True, True, False]})
    result = Cube(measures=[Count(where=Col("v") > 10), Sum("v", where=Col("v") > 10)]).compute(
        data
    )
    assert result.layout().iloc[0].tolist() == [2, 50]
    data.loc[3, "v"] = np.inf
    assert (
        Cube(measures=[Sum("v", where=Col("keep").isin([True]))]).compute(data).layout().iloc[0, 0]
        == 55
    )


def test_same_predicate_shared_between_count_and_sum():
    from dataclasses import dataclass

    from phl_risk.analysis._expressions import Predicate

    calls = []

    @dataclass(frozen=True)
    class Positive(Predicate):
        def required_columns(self):
            return ("v",)

        def evaluate(self, data, *, backend="pandas"):
            calls.append(len(data))
            return data["v"] > 0

    condition = Positive()
    result = Cube(["g"], [Count(where=condition), Sum("v", where=condition)]).compute(
        pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, -2.0, 3.0]})
    )
    assert result.layout().to_numpy().tolist() == [[1, 1], [1, 3]]
    assert calls == [3]
