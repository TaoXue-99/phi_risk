"""Small backend boundary for batched two-population calculations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ComparisonGroups:
    """Shared row-to-group encoding; calculators must treat arrays as read-only."""

    reference_codes: NDArray[np.intp]
    current_codes: NDArray[np.intp]
    size: int


@dataclass(frozen=True)
class ComparisonOutput:
    values: NDArray[np.float64]
    metadata: dict[str, object] = field(default_factory=dict)


class ComparativeCalculation(ABC):
    """One vectorized calculation across aligned groups, with no layout knowledge.

    Input frames are read-only by contract. Implementations declare backend
    support; future raw-sample comparisons need not build distribution profiles.
    """

    @abstractmethod
    def required_columns(self) -> tuple[str, ...]: ...

    @abstractmethod
    def evaluate(
        self,
        reference: object,
        current: object,
        groups: ComparisonGroups,
        *,
        backend: str,
        on_invalid: str,
    ) -> ComparisonOutput: ...
