"""Small immutable predicate vocabulary, evaluated with pandas vector operations."""

from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Hashable

import pandas as pd

from phl_risk.exceptions import EngineError, MeasureError


class Predicate:
    def required_columns(self) -> tuple[str, ...]:
        raise NotImplementedError

    def evaluate(self, data: object, *, backend: str = "pandas") -> pd.Series:
        raise NotImplementedError

    def __and__(self, other: "Predicate") -> "Predicate":
        return Logical("and", self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return Logical("or", self, other)

    def __invert__(self) -> "Predicate":
        return Logical("not", self)

    def __bool__(self):
        raise TypeError("Use &, | and ~ with parenthesized predicates, not and/or/not")


@dataclass(frozen=True)
class Comparison(Predicate):
    column: str
    operation: str
    value: Hashable = None

    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def evaluate(self, data, *, backend="pandas"):
        if backend != "pandas" or not isinstance(data, pd.DataFrame):
            raise EngineError("Predicates currently support the pandas backend")
        if self.column not in data:
            raise EngineError(f"Missing predicate column {self.column!r}")
        values = data[self.column]
        if self.operation == "isna":
            return values.isna()
        if self.operation == "notna":
            return values.notna()
        if self.operation == "isin":
            result = values.isin(self.value)
        else:
            result = {"eq": eq, "ne": ne, "gt": gt, "ge": ge, "lt": lt, "le": le}[self.operation](
                values, self.value
            )
        # Unknown remains unknown through negation and boolean composition.
        return result.astype("boolean").where(values.notna(), pd.NA)


@dataclass(frozen=True)
class Logical(Predicate):
    operation: str
    left: Predicate
    right: Predicate | None = None

    def __post_init__(self):
        if not isinstance(self.left, Predicate) or (
            self.operation != "not" and not isinstance(self.right, Predicate)
        ):
            raise MeasureError("Logical operands must be predicates")

    def required_columns(self):
        right = () if self.right is None else self.right.required_columns()
        return tuple(dict.fromkeys(self.left.required_columns() + right))

    def evaluate(self, data, *, backend="pandas"):
        left = self.left.evaluate(data, backend=backend)
        if self.operation == "not":
            return ~left
        right = self.right.evaluate(data, backend=backend)
        return left & right if self.operation == "and" else left | right


def _literal(value):
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise MeasureError("Comparison values must be non-missing scalars; use isna()/notna()")
    try:
        hash(value)
    except TypeError as exc:
        raise MeasureError("Comparison values must be hashable") from exc
    return value


@dataclass(frozen=True, eq=False)
class Col:
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise MeasureError("Col requires a non-empty column name")

    def __eq__(self, value):
        return Comparison(self.name, "eq", _literal(value))

    def __ne__(self, value):
        return Comparison(self.name, "ne", _literal(value))

    def __gt__(self, value):
        return Comparison(self.name, "gt", _literal(value))

    def __ge__(self, value):
        return Comparison(self.name, "ge", _literal(value))

    def __lt__(self, value):
        return Comparison(self.name, "lt", _literal(value))

    def __le__(self, value):
        return Comparison(self.name, "le", _literal(value))

    def isin(self, values):
        if isinstance(values, (str, bytes)):
            raise MeasureError("isin requires a sequence of scalar values")
        return Comparison(self.name, "isin", tuple(_literal(v) for v in values))

    def isna(self):
        return Comparison(self.name, "isna")

    def notna(self):
        return Comparison(self.name, "notna")
