"""Backend-neutral execution vocabulary, private in V0.1."""

from dataclasses import dataclass
from typing import Hashable, Literal

from ._expressions import Predicate


@dataclass(frozen=True)
class AggregateNode:
    operation: Literal["row_count", "valid_count", "event_count", "sum", "count_where"]
    column: str | None = None
    event_value: Hashable = 1
    weight: str | None = None
    condition: Predicate | None = None
    missing: str = "propagate"

    def required_columns(self) -> tuple[str, ...]:
        columns = tuple(x for x in (self.column, self.weight) if x is not None)
        return columns + (() if self.condition is None else self.condition.required_columns())


@dataclass(frozen=True)
class GroupMetricNode:
    kernel: Literal["auc", "ks"]
    target: str
    score: str
    weight: str | None = None

    def required_columns(self) -> tuple[str, ...]:
        return tuple(x for x in (self.target, self.score, self.weight) if x is not None)


@dataclass(frozen=True)
class DerivedMetricNode:
    numerator: AggregateNode
    denominator: AggregateNode | Literal["all"]

    def required_columns(self) -> tuple[str, ...]:
        other = () if self.denominator == "all" else self.denominator.required_columns()
        return tuple(dict.fromkeys(self.numerator.required_columns() + other))


@dataclass(frozen=True)
class RatioNode:
    numerator: str
    denominator: str

    def required_columns(self) -> tuple[str, ...]:
        return ()  # References measure names, resolved by CubePlan.


Node = AggregateNode | GroupMetricNode | DerivedMetricNode | RatioNode


@dataclass(frozen=True)
class MeasureSpec:
    name: str
    node: Node
    # Empty cells are absent from the sparse result, materialized by Layout.
    empty_value: float = float("nan")

    def required_columns(self) -> tuple[str, ...]:
        return self.node.required_columns()
