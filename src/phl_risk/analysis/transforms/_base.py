"""Row-preserving transformer contract; independent of Cube and grouping."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class BaseTransformer(ABC, Generic[Input, Output]):
    """Fit learns state; transform must never learn from its input."""

    @abstractmethod
    def fit(self, X: Input, y: object = None) -> "BaseTransformer[Input, Output]":
        """Learn reference state and return self."""

    @abstractmethod
    def transform(self, X: Input) -> Output:
        """Apply learned state without modifying X."""

    def fit_transform(self, X: Input, y: object = None) -> Output:
        return self.fit(X, y).transform(X)

    @property
    def is_fitted(self) -> bool:
        return False

    def metadata(self) -> dict[str, object]:
        return {"transformer": type(self).__name__, "fitted": self.is_fitted}
