from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import (
    AUC,
    KS,
    AnalysisContext,
    BaseCubeEngine,
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
from phl_risk.exceptions import EngineError, InvalidMetricError, NotFittedError, TransformError
from phl_risk.metrics import auc_score, event_rate, ks_score


@pytest.mark.parametrize("dimensions", [[], ["dt"], ["dt", "user_type"]])
def test_global_one_two_dimensions(fixture_analysis_df, dimensions):
    df = fixture_analysis_df
    original = df.copy(deep=True)
    cube = Cube(
        dimensions,
        [Count(), Share(), EventRate("label"), AUC("score", "label"), KS("score", "label")],
    )
    result = cube.compute(df)
    assert list(result.data_.columns) == dimensions + ["metric", "value"]
    assert result.ndim == len(dimensions) + 1
    assert result.axes[-1].values == (
        "count",
        "share",
        "event_rate__label__1",
        "auc__score",
        "ks__score",
    )
    table = result.layout()
    groups = [(0, df)] if not dimensions else df.groupby(dimensions, sort=False)
    for key, group in groups:
        if len(dimensions) == 1:
            key = key[0] if isinstance(key, tuple) else key
        assert table.loc[key, "count"] == len(group)
        assert table.loc[key, "share"] == pytest.approx(len(group) / len(df))
        assert table.loc[key, "event_rate__label__1"] == pytest.approx(event_rate(group.label))
        assert table.loc[key, "auc__score"] == pytest.approx(auc_score(group.label, group.score))
        assert table.loc[key, "ks__score"] == pytest.approx(ks_score(group.label, group.score))
    pd.testing.assert_frame_equal(df, original)


def test_regression_fixture(fixture_analysis_df):
    result = Cube(["dt", "user_type"], [AUC("score", "label"), KS("score", "label")]).compute(
        fixture_analysis_df
    )
    np.testing.assert_allclose(result.layout().to_numpy(), [[0.75, 0.5], [1, 1], [0, 1], [0.5, 0]])


def test_share_is_not_event_rate():
    df = pd.DataFrame({"group": ["cell"] * 10 + ["rest"] * 90, "label": [1] * 3 + [0] * 97})
    result = Cube(["group"], [Count(), Share(), EventRate("label")]).compute(df)
    assert result.layout().loc["cell"].tolist() == [10, 0.1, 0.3]
    assert result.metadata_["shared_aggregates"] == 3
    assert result.metadata_["native_aggregation_passes"] == 1


def test_weights_and_explicit_context_override():
    df = pd.DataFrame(
        {
            "g": ["a"] * 4,
            "y": [0, 1, None, 1],
            "other": [1, 0, 1, 0],
            "s": [0.1, 0.9, 0.5, 0.2],
            "w": [1, 2, 99, 4],
        }
    )
    context = AnalysisContext(target="y", weight="w")
    measures = [
        Count(),
        Share(),
        EventRate(),
        EventRate("other", name="override"),
        AUC("s"),
        KS("s"),
    ]
    result = Cube(["g"], measures).compute(df, context)
    row = result.layout().loc["a"]
    assert row["count"] == 4 and row["share"] == 1
    assert row["event_rate__y__1"] == pytest.approx(6 / 7)
    assert row["override"] == pytest.approx(100 / 106)
    assert row["auc__s"] == pytest.approx(auc_score(df.y, df.s, df.w))
    assert row["ks__s"] == pytest.approx(ks_score(df.y, df.s, df.w))
    with pytest.raises(ValueError, match="finite non-negative"):
        Cube(measures=[EventRate("y", weight="w")]).compute(df.assign(w=-1))


@pytest.mark.parametrize("dimension", ["drop", "keep"])
def test_missing_policy_and_metric_local_deletion(dimension):
    df = pd.DataFrame(
        {"g": ["a", "a", None, None], "y": [1, None, 0, 1], "s": [None, 0.2, 0.1, 0.9]}
    )
    result = Cube(
        ["g"],
        [Count(), Share(), EventRate("y"), AUC("s", "y")],
        policy=ComputePolicy(MissingPolicy(dimension=dimension)),
    ).compute(df)
    table = result.layout()
    assert table.loc["a", "count"] == 2
    assert table.loc["a", "event_rate__y__1"] == 1
    assert np.isnan(table.loc["a", "auc__s"])
    assert result.metadata_["analyzed_rows"] == (2 if dimension == "drop" else 4)
    assert table["share"].sum() == 1
    assert result.shape[0] == (1 if dimension == "drop" else 2)


@pytest.mark.parametrize("dims", [[], ["g"], ["g", "h"]])
def test_empty_data(dims):
    df = pd.DataFrame(
        {
            "g": pd.Series(dtype=str),
            "h": pd.Series(dtype=str),
            "y": pd.Series(dtype=float),
            "s": pd.Series(dtype=float),
        }
    )
    result = Cube(dims, [Count(), Share(), EventRate("y"), AUC("s", "y")]).compute(df)
    table = result.layout()
    if dims:
        assert table.empty and result.data_.empty
    else:
        assert table.loc[0, "count"] == 0
        assert table.iloc[0, 1:].isna().all()


def test_invalid_policy():
    df = pd.DataFrame({"g": ["a"], "y": [1], "s": [0.5]})
    with pytest.warns(RuntimeWarning):
        Cube(["g"], [AUC("s", "y")], policy=ComputePolicy(on_invalid="warn")).compute(df)
    with pytest.raises(InvalidMetricError, match="group"):
        Cube(["g"], [AUC("s", "y")], policy=ComputePolicy(on_invalid="raise")).compute(df)
    with pytest.raises(InvalidMetricError, match="zero denominator"):
        Cube(measures=[EventRate("y")], policy=ComputePolicy(on_invalid="raise")).compute(
            df.assign(y=None)
        )


def test_filters_and_input_preservation():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [0, 1, 2, 3]}, index=[2, 2, 1, 1])
    original = df.copy(deep=True)

    def predicate(frame):
        frame["unwanted"] = 1
        return frame.x > 0

    result = Cube(["g"], [Count(), Share()], filters=[predicate]).compute(df)
    assert result.layout().loc["a", "share"] == pytest.approx(1 / 3)
    assert result.metadata_["filtered_rows"] == 3
    pd.testing.assert_frame_equal(df, original)
    with pytest.raises(EngineError, match="boolean mask"):
        Cube(filters=[lambda frame: frame.x]).compute(df)

    @dataclass(frozen=True)
    class Positive:
        def required_columns(self):
            return ("x",)

        def evaluate(self, data, *, backend):
            assert backend == "pandas"
            return data.x > 0

    assert Cube(filters=[Positive()]).compute(df).layout().iloc[0, 0] == 3


def test_reference_lifecycle_isolation_snapshot_and_atomic_refit():
    reference = pd.DataFrame({"x": range(10), "y": range(10)})
    dim = BinDimension("x", QuantileBinner(2))
    cube = Cube([dim, BinDimension("y", QuantileBinner(2))])
    other = Cube([dim])
    assert not cube.is_fitted and "False" in cube.explain(format="text")
    with pytest.raises(NotFittedError):
        cube.compute(reference)
    assert cube.fit(reference) is cube
    old_plan = cube.plan()
    old_result = cube.compute(reference)
    assert cube.is_fitted and not other.is_fitted and not dim.is_fitted
    exposed = cube.dimensions_
    exposed[0].fit(reference.assign(x=100))
    assert cube.dimensions_[0].transformer.bin_edges_[1] == 4.5
    with pytest.raises(TransformError):
        cube.fit(reference.assign(x=100, y=None))
    pd.testing.assert_frame_equal(old_result.data_, cube.compute(reference).data_)
    cube.fit(reference.assign(x=reference.x + 100, y=reference.y + 100))
    assert old_plan.dimensions[0].transformer.bin_edges_[1] == 4.5
    pd.testing.assert_frame_equal(
        PandasEngine().execute(old_plan, reference).data_, old_result.data_
    )


def test_fit_filters_and_missing_columns(fixture_analysis_df):
    frame = pd.DataFrame({"x": [0, 1, 2, 3]})
    cube = Cube([BinDimension("x", QuantileBinner(2))], filters=[lambda df: df.x > 1])
    assert cube.fit(frame).dimensions_[0].transformer.bin_edges_[1] == 2.5
    with pytest.raises(EngineError, match="Missing required columns"):
        Cube(measures=[AUC("absent", "label")]).compute(fixture_analysis_df)
    with pytest.raises(EngineError, match="duplicate"):
        Cube().compute(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(EngineError, match="DataFrame"):
        Cube().compute([1, 2])
    with pytest.raises(EngineError, match="Unsupported engine"):
        Cube(engine="polars")


def test_engine_dispatch(fixture_analysis_df):
    class RecordingEngine(BaseCubeEngine):
        name = "recording"

        def execute(self, plan, data, context=None):
            self.received = plan
            return PandasEngine().execute(plan, data, context)

    backend = RecordingEngine()
    result = Cube(["dt"]).compute(fixture_analysis_df, engine=backend)
    assert backend.received.group_columns == ("dt",)
    assert result.layout()["count"].sum() == len(fixture_analysis_df)


def test_extreme_weights_scale_within_valid_group():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b"],
            "y": [0, 1, None, 0, 1],
            "w": [1e-308, 1e-308, 1e308, 1e308, 1e308],
        }
    )
    result = Cube(["g"], [EventRate("y", weight="w")]).compute(df)
    np.testing.assert_allclose(result.layout().to_numpy(), [[0.5], [0.5]])
