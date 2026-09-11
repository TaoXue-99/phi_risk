"""Lightweight logical plan; contains no data or backend expressions."""

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

from phl_risk.exceptions import MeasureError

from ._context import AnalysisContext, ComputePolicy
from ._nodes import (
    AggregateNode,
    ComparativeNode,
    DerivedMetricNode,
    GroupMetricNode,
    MeasureSpec,
    RatioNode,
)
from .dimensions import BaseDimension


@runtime_checkable
class FilterExpression(Protocol):
    """Extension hook for expressions with backend-specific compilation."""

    def required_columns(self) -> tuple[str, ...]: ...

    def evaluate(self, data: object, *, backend: str) -> object: ...


FilterLike = FilterExpression | Callable[[object], object]


@dataclass(frozen=True)
class CubePlan:
    dimensions: tuple[BaseDimension, ...]
    measures: tuple[MeasureSpec, ...]
    filters: tuple[FilterLike, ...] = ()
    context: AnalysisContext = AnalysisContext()
    policy: ComputePolicy = ComputePolicy()
    totals: bool = False

    @property
    def mode(self) -> str:
        modes = {isinstance(m.node, ComparativeNode) for m in self.measures}
        if len(modes) > 1:
            raise MeasureError("Cannot mix single-sample and comparative measures")
        return "comparative" if True in modes else "single"

    def __post_init__(self):
        self.mode
        self.evaluation_measures  # Validate references even when only explaining a plan.

    @property
    def evaluation_measures(self) -> tuple[MeasureSpec, ...]:
        by_name = {m.name: m for m in self.measures}
        if len(by_name) != len(self.measures):
            raise MeasureError("Measure names must be unique")
        ordered, active, done = [], set(), set()

        def visit(name):
            if name not in by_name:
                raise MeasureError(f"Unknown measure reference {name!r}")
            if name in active:
                raise MeasureError(f"Cyclic measure reference involving {name!r}")
            if name in done:
                return
            active.add(name)
            measure = by_name[name]
            if isinstance(measure.node, RatioNode):
                visit(measure.node.numerator)
                visit(measure.node.denominator)
            active.remove(name)
            done.add(name)
            ordered.append(measure)

        for name in by_name:
            visit(name)
        return tuple(ordered)

    @property
    def group_columns(self) -> tuple[str, ...]:
        return tuple(d.output_name for d in self.dimensions)

    def required_columns(self) -> tuple[str, ...]:
        columns = [c for d in self.dimensions for c in d.required_columns()]
        columns.extend(c for m in self.measures for c in m.required_columns())
        for predicate in self.filters:
            if isinstance(predicate, FilterExpression):
                columns.extend(predicate.required_columns())
        return tuple(dict.fromkeys(columns))

    @property
    def aggregates(self) -> tuple[AggregateNode, ...]:
        nodes = []
        for measure in self.measures:
            node = measure.node
            if isinstance(node, AggregateNode):
                nodes.append(node)
            elif isinstance(node, DerivedMetricNode):
                nodes.append(node.numerator)
                if isinstance(node.denominator, AggregateNode):
                    nodes.append(node.denominator)
        return tuple(dict.fromkeys(nodes))

    @property
    def group_metrics(self) -> tuple[GroupMetricNode, ...]:
        return tuple(
            dict.fromkeys(m.node for m in self.measures if isinstance(m.node, GroupMetricNode))
        )

    def explain(self, engine: str = "pandas", *, format: Literal["table", "text"] = "table"):
        from ._explain import explain_plan

        return explain_plan(self, engine, format)
