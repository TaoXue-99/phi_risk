"""Partition assignment declarations; no splitter executes on real data."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal, TypeAlias

from phl_risk.exceptions import PlanError

from .._utils import Columns, JSONValue, columns, name

PartitionValue: TypeAlias = str | int | float | bool


def _seed(value: int) -> None:
    if type(value) is not int or value < 0:
        raise PlanError("random_state must be a non-negative integer")


@dataclass(frozen=True)
class BaseSplitter(ABC):
    """Describe assignment rules and the partition definition mode they accept."""

    @property
    @abstractmethod
    def partition_mode(self) -> Literal["ratios", "values"]:
        """Interpretation required for the accompanying PartitionSpec values."""

    @abstractmethod
    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize assignment rules, never assign observations."""


@dataclass(frozen=True, init=False)
class HashSplitter(BaseSplitter):
    """Declare stable group assignment by explicit keys, independent of sample_key."""

    key: tuple[str, ...]
    random_state: int
    partition_mode: ClassVar[Literal["ratios"]] = "ratios"

    def __init__(self, key: Columns, random_state: int = 2026) -> None:
        object.__setattr__(self, "key", columns(key, "HashSplitter key"))
        _seed(random_state)
        object.__setattr__(self, "random_state", random_state)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize hash keys and seed; hashing is deferred to execution."""
        return {"type": "HashSplitter", "key": list(self.key), "random_state": self.random_state}


@dataclass(frozen=True)
class RandomSplitter(BaseSplitter):
    """Declare seeded random assignment, optionally stratified by one column."""

    random_state: int = 2026
    stratify: str | None = None
    partition_mode: ClassVar[Literal["ratios"]] = "ratios"

    def __post_init__(self) -> None:
        _seed(self.random_state)
        if self.stratify is not None:
            name(self.stratify, "RandomSplitter stratify")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize random assignment intent without generating randomness."""
        return {
            "type": "RandomSplitter",
            "random_state": self.random_state,
            "stratify": self.stratify,
        }


@dataclass(frozen=True)
class ColumnSplitter(BaseSplitter):
    """Assign partition names to existing scalar values of a source column."""

    column: str
    partition_mode: ClassVar[Literal["values"]] = "values"

    def __post_init__(self) -> None:
        name(self.column, "ColumnSplitter column")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the source partition column."""
        return {"type": "ColumnSplitter", "column": self.column}


@dataclass(frozen=True)
class TimeSplitter(BaseSplitter):
    """Declare ascending-time ratio partitions in PartitionSpec insertion order.

    Only explicit time columns are supported. Boundaries, ties, missing timestamps
    and rounding require an execution policy in a later phase; no dates are filtered.
    """

    time: str
    partition_mode: ClassVar[Literal["ratios"]] = "ratios"

    def __post_init__(self) -> None:
        name(self.time, "TimeSplitter time")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize time ordering intent without sorting data."""
        return {"type": "TimeSplitter", "time": self.time}


@dataclass(frozen=True, init=False, eq=False)
class PartitionSpec:
    """Ordered partition names mapped to ratios or source-column scalar values.

    Definitions are copied and read-only. Their interpretation belongs to SplitSpec,
    allowing numeric ColumnSplitter labels without mistaking them for ratios.
    """

    definitions: Mapping[str, PartitionValue]

    def __init__(self, **partitions: PartitionValue) -> None:
        if "train" not in partitions:
            raise PlanError("Partitions must include 'train'")
        normalized = {}
        for partition, value in partitions.items():
            name(partition, "partition name")
            if type(value) not in (str, int, float, bool):
                raise PlanError(f"Partition {partition!r} requires a JSON scalar value")
            if isinstance(value, str):
                name(value, f"Partition {partition!r} value")
            if isinstance(value, float) and not math.isfinite(value):
                raise PlanError(f"Partition {partition!r} value must be finite")
            normalized[partition] = value
        object.__setattr__(self, "definitions", MappingProxyType(normalized))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartitionSpec):
            return NotImplemented
        return tuple(self.definitions.items()) == tuple(other.definitions.items())

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a detached mapping preserving partition declaration order."""
        return dict(self.definitions)


@dataclass(frozen=True)
class SplitSpec:
    """Combine assignment rules with independently named dataset partitions."""

    splitter: BaseSplitter
    partitions: PartitionSpec

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check splitter/partition mode compatibility without accessing data."""
        if not isinstance(self.splitter, BaseSplitter):
            raise PlanError("SplitSpec splitter must be a BaseSplitter declaration")
        if not isinstance(self.partitions, PartitionSpec):
            raise PlanError("SplitSpec partitions must be a PartitionSpec")
        values = tuple(self.partitions.definitions.values())
        if self.splitter.partition_mode == "ratios":
            for key, value in self.partitions.definitions.items():
                if type(value) not in (int, float) or not 0 < value <= 1:
                    raise PlanError(f"Partition {key!r} ratio must be numeric and in (0, 1]")
            if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise PlanError("Partition ratios must sum to 1 (tolerance 1e-9)")
        elif self.splitter.partition_mode == "values":
            if len(set(values)) != len(values):
                raise PlanError("ColumnSplitter source values must be distinct across partitions")
        else:
            raise PlanError(f"Unsupported partition mode: {self.splitter.partition_mode!r}")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize assignment rules, mode and ordered partition definitions."""
        return {
            "splitter": self.splitter.to_dict(),
            "partition_mode": self.splitter.partition_mode,
            "partitions": self.partitions.to_dict(),
        }
