"""Lightweight logical plan; contains no data or backend expressions."""

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

from ._context import AnalysisContext, ComputePolicy
from ._nodes import AggregateNode, DerivedMetricNode, GroupMetricNode, MeasureSpec
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
