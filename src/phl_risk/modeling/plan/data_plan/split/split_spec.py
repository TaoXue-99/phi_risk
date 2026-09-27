"""Combine assignment rules and partitions, validating their shared contract."""

import math
from dataclasses import dataclass

from phl_risk.exceptions import PlanError
from phl_risk.modeling._utils import JSONValue

from .partition import PartitionSpec
from .splitter import BaseSplitter


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
