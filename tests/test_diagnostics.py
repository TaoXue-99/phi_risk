import numpy as np
import pandas as pd

from phl_risk.analysis import AUC, Count, Cube, Funnel, Sum


def test_sum_funnel_diagnostics_and_totals_preserve_values():
    df = pd.DataFrame({"g": ["a"] * 1000, "before": [1.0] * 1000, "after": [1.0] * 999 + [np.nan]})
    result = Cube(["g"], Funnel(["before", "after"]).measures()).compute(df, totals=True)
    before = result.data_
    report = result.diagnostics()
    row = report.loc[("a", "after数量")]
    assert row["reason"] == "missing_propagated"
    assert row["field"] == "after"
    assert row["affected_rows"] == 1
    assert row["input_rows"] == 1000
    assert report.loc[("a", "before→after转化率"), "dependency"] == "after数量"
    assert result.total(over=["g"]).diagnostics().loc["after数量", "affected_rows"] == 1
    report.loc[("a", "after数量"), "message"] = "changed"
    assert result.diagnostics().loc[("a", "after数量"), "message"] != "changed"
    pd.testing.assert_frame_equal(before, result.data_)
    assert np.isnan(result.layout().loc["a", "after数量"])


def test_empty_report_and_uninstrumented_nan_are_distinguished():
    data = pd.DataFrame({"g": ["a", "a"], "y": [1, 1], "score": [0.1, 0.2]})
    assert Cube(["g"], [Count()]).compute(data).diagnostics().empty
    report = Cube(["g"], [AUC(score="score", target="y")]).compute(data).diagnostics()
    assert report.loc[("a", "auc__score"), "reason"] == "reason_not_recorded"


def test_multiple_dimensions_missing_key_and_no_layout_only_rows():
    data = pd.DataFrame({"g": ["a", "b"], "h": ["x", "y"], "v": [np.nan, 1.0]})
    result = Cube(["g", "h"], [Sum("v")]).compute(data)
    report = result.diagnostics()
    assert report.index.names == ["g", "h", "metric"]
    assert report.index.tolist() == [("a", "x", "sum__v")]


def test_zero_denominator_and_missing_dimension():
    from phl_risk.analysis import ComputePolicy, MissingPolicy, Ratio

    data = pd.DataFrame({"reason": [None], "a": [0.0], "b": [0.0]})
    result = Cube(
        ["reason"],
        [Sum("a", name="a"), Sum("b", name="b"), Ratio("a", "b", name="rate")],
        policy=ComputePolicy(missing=MissingPolicy(dimension="keep")),
    ).compute(data)
    report = result.diagnostics()
    assert report.index.names == ["reason", "metric"]
    assert report.iloc[0]["reason"] == "zero_denominator"
    assert pd.isna(report.index[0][0])


def test_empty_population_report_schema():
    result = Cube(["g"], [Count()]).compute(pd.DataFrame({"g": []}))
    assert result.diagnostics().empty
    assert result.diagnostics().index.names == ["g", "metric"]
    assert str(result.diagnostics()["affected_rows"].dtype) == "Int64"
