"""Backend-neutral execution vocabulary, private in V0.1."""

from dataclasses import dataclass
from typing import Hashable, Literal


@dataclass(frozen=True)
class AggregateNode:
    operation: Literal["row_count", "valid_count", "event_count"]
    column: str | None = None
    event_value: Hashable = 1
    weight: str | None = None

    def required_columns(self) -> tuple[str, ...]:
        return tuple(x for x in (self.column, self.weight) if x is not None)


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


Node = AggregateNode | GroupMetricNode | DerivedMetricNode


@dataclass(frozen=True)
class MeasureSpec:
    name: str
    node: Node
    # Empty cells are absent from the sparse result, materialized by Layout.
    empty_value: float = float("nan")

    def required_columns(self) -> tuple[str, ...]:
        return self.node.required_columns()
