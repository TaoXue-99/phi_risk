from dataclasses import dataclass

import pytest

from phl_risk.analysis import ComparativeMeasure, Count, Cube, PandasEngine, SingleSampleMeasure
from phl_risk.analysis._comparison_types import ComparativeCalculation
from phl_risk.analysis._nodes import ComparativeNode, MeasureSpec
from phl_risk.exceptions import EngineError, MeasureError


class PlaceholderCalculation(ComparativeCalculation):
    def required_columns(self):
        return ()

    def evaluate(self, reference, current, groups, *, backend, on_invalid):
        raise NotImplementedError


@dataclass(frozen=True)
class Placeholder(ComparativeMeasure):
    def compile(self, context=None):
        return MeasureSpec("comparison", ComparativeNode(PlaceholderCalculation()))


def test_modes_and_compatibility():
    assert isinstance(Count(), SingleSampleMeasure)
    assert Cube(measures=[Count()]).plan().mode == "single"
    cube = Cube(measures=[Placeholder()])
    assert cube.plan().mode == "comparative"
    with pytest.raises(MeasureError, match="compute_comparison"):
        cube.compute(None)
    with pytest.raises(MeasureError, match="compute_comparison"):
        cube.fit_compute(None)
    with pytest.raises(MeasureError, match="Cube.compute"):
        Cube(measures=[Count()]).compute_comparison(None, None)
    with pytest.raises(MeasureError, match="mix"):
        Cube(measures=[Count(), Placeholder()])
    with pytest.raises(EngineError, match="execute_comparison"):
        PandasEngine().execute(cube.plan(), None)
