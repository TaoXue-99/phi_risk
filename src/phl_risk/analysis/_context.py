from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AnalysisContext:
    """Explicit defaults; measure configuration takes precedence."""

    target: str | None = None
    weight: str | None = None

    def __post_init__(self) -> None:
        for name in ("target", "weight"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"AnalysisContext.{name} must be a non-empty string or None")


@dataclass(frozen=True)
class MissingPolicy:
    """Dimension drop/keep; target and score deletion is local to each metric."""

    dimension: Literal["drop", "keep"] = "drop"
    target: Literal["drop"] = "drop"
    score: Literal["drop"] = "drop"

    def __post_init__(self) -> None:
        if self.dimension not in ("drop", "keep"):
            raise ValueError("dimension missing policy must be 'drop' or 'keep'")
        if self.target != "drop" or self.score != "drop":
            raise ValueError("V0.1 supports only per-metric target/score deletion")


@dataclass(frozen=True)
class ComputePolicy:
    missing: MissingPolicy = MissingPolicy()
    on_invalid: Literal["nan", "warn", "raise"] = "nan"

    def __post_init__(self) -> None:
        if not isinstance(self.missing, MissingPolicy):
            raise TypeError("missing must be MissingPolicy")
        if self.on_invalid not in ("nan", "warn", "raise"):
            raise ValueError("on_invalid must be 'nan', 'warn', or 'raise'")
