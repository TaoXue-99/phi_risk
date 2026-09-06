from dataclasses import FrozenInstanceError, dataclass

import pandas as pd
import pytest

from phl_risk.analysis import AUC, KS, AnalysisContext, BaseMeasure, Count, Cube, EventRate, Share
from phl_risk.analysis._nodes import AggregateNode, DerivedMetricNode, MeasureSpec
from phl_risk.exceptions import DimensionError, MeasureError


def test_dependencies_names_context():
    ctx = AnalysisContext(target="default_label", weight="w")
    assert AUC("s").compile(ctx).name == "auc__s"
    assert KS("s").required_columns(ctx) == ("default_label", "s", "w")
    assert AUC("s", "override", weight="v").required_columns(ctx) == ("override", "s", "v")
    assert (
        EventRate(event_value="SUCCESS").compile(ctx).name == "event_rate__default_label__SUCCESS"
    )
    assert Count().required_columns(ctx) == ()
    assert Share().required_columns(ctx) == ()
    with pytest.raises(MeasureError, match="target"):
        AUC("s").compile()


def test_shared_native_aggregations():
    cube = Cube(["dt"], [Count(), Share(), EventRate("label"), EventRate("label", name="again")])
    plan = cube.plan()
    assert len(plan.aggregates) == 3
    assert plan.aggregates.count(AggregateNode("row_count")) == 1
    assert isinstance(plan.measures[1].node, DerivedMetricNode)
    assert plan.required_columns() == ("dt", "label")
    with pytest.raises(FrozenInstanceError):
        plan.context = AnalysisContext()
    explanation = cube.explain(format="text")
    for word in ("dt", "event_rate__label__1", "pandas", "Shared aggregates"):
        assert word in explanation
    table = cube.explain()
    assert table.loc[table.Item == "Shared aggregates", "Description"].iloc[0] == "3"


def test_name_collisions_and_unsupported_scopes():
    with pytest.raises(DimensionError, match="Duplicate"):
        Cube(["dt", "dt"])
    with pytest.raises(MeasureError, match="Duplicate"):
        Cube(measures=[Count(), Count()]).plan()
    with pytest.raises(MeasureError, match="Duplicate"):
        Cube(measures=[AUC("s", "a"), AUC("s", "b")]).plan()
    with pytest.raises(MeasureError, match="denominator"):
        Share("row")
    with pytest.raises(MeasureError):
        Cube(measures=[])


def test_measure_extension_uses_existing_nodes():
    @dataclass(frozen=True)
    class SampleCount(BaseMeasure):
        def compile(self, context=None):
            return MeasureSpec("samples", AggregateNode("row_count"), 0)

    result = Cube(measures=[SampleCount()]).compute(pd.DataFrame({"x": [1, 2, 3]}))
    assert result.layout().loc[0, "samples"] == 3
