from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._plan import CubePlan, FilterLike

if TYPE_CHECKING:
    from phl_risk.analysis._result import CubeResult


class BaseCubeEngine(ABC):
    """Backend computation, independent of task scheduling or parallel execution."""

    name = "base"

    def prepare_fit(self, data: object, filters: tuple[FilterLike, ...]) -> object:
        """Apply reference filters before fitting dimension transformers."""
        if filters:
            raise NotImplementedError(f"{self.name} does not support fit filters")
        return data

    def prepare_fit_with_positions(self, data: object, filters: tuple[FilterLike, ...]):
        """Optional positional alignment for supervised stratified fitting.

        Backends that filter rows must return their original integer positions,
        rather than relying on input index labels being unique.
        """
        if filters:
            raise NotImplementedError(f"{self.name} does not support positional fit filtering")
        return self.prepare_fit(data, filters), np.arange(len(data))

    @abstractmethod
    def execute(
        self, plan: CubePlan, data: object, context: AnalysisContext | None = None
    ) -> "CubeResult":
        """Execute a resolved plan and return a canonical result."""

    def execute_comparison(
        self,
        plan: CubePlan,
        reference: object,
        current: object,
    ) -> "CubeResult":
        """Optional backend capability; existing single-sample engines still work."""
        raise NotImplementedError(f"{self.name} does not support comparative execution")
