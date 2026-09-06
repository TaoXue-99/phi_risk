from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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

    @abstractmethod
    def execute(
        self, plan: CubePlan, data: object, context: AnalysisContext | None = None
    ) -> "CubeResult":
        """Execute a resolved plan and return a canonical result."""
