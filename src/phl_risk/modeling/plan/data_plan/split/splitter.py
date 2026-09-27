"""Rules declaring future observation assignment; never execute a split."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from phl_risk.exceptions import PlanError
from phl_risk.modeling._utils import Columns, JSONValue, columns, name


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
