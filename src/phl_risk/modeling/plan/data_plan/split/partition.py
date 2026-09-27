"""Ordered partition names and their ratio or source-value definitions."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from phl_risk.exceptions import PlanError
from phl_risk.modeling._utils import JSONValue, name

PartitionValue: TypeAlias = str | int | float | bool


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
