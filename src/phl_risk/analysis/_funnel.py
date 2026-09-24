"""Funnel configuration expands into ordinary measures; it never processes data."""

from dataclasses import dataclass, replace
from typing import Sequence

from phl_risk.exceptions import MeasureError

from .measures import Count, Ratio, Sum
from .measures._base import measure_name


@dataclass(frozen=True)
class Stage:
    """A named stage backed by a quantity measure; Funnel assigns its output name."""

    name: str
    measure: Sum | Count

    def __post_init__(self):
        measure_name(self.name, "")
        if not isinstance(self.measure, (Sum, Count)):
            raise MeasureError("Stage requires Sum or Count")


@dataclass(frozen=True)
class Transition:
    """Conversion from before to after: sum(after) / sum(before)."""

    before: str
    after: str
    name: str | None = None

    def __post_init__(self):
        measure_name(self.before, "")
        measure_name(self.after, "")
        measure_name(self.name, "conversion")


@dataclass(frozen=True)
class Funnel:
    """Ordered, nested stages with a configurable number of conversion measures.

    String stages name source columns and use Sum. Explicit Stage objects allow
    conditional counts. Rates are adjacent, from_first, both, or a sequence of
    forward Transition objects. No row-level nesting or deduplication is inferred.
    """

    stages: Sequence[str | Stage]
    aggregation: str = "sum"
    rates: str | Sequence[Transition] = "adjacent"

    def __post_init__(self):
        if self.aggregation != "sum":
            raise MeasureError("aggregation must be 'sum'; use Stage for conditional counts")
        if isinstance(self.stages, (str, bytes)):
            raise MeasureError("stages must be a sequence, not a single string")
        stages = tuple(Stage(s, Sum(s)) if isinstance(s, str) else s for s in self.stages)
        if not stages or any(not isinstance(s, Stage) for s in stages):
            raise MeasureError("Funnel requires at least one string or Stage")
        if len({s.name for s in stages}) != len(stages):
            raise MeasureError("Stage names must be unique")
        object.__setattr__(self, "stages", stages)
        if isinstance(self.rates, str):
            if self.rates not in ("adjacent", "from_first", "both"):
                raise MeasureError("rates must be adjacent, from_first, both, or transitions")
        else:
            object.__setattr__(self, "rates", tuple(self.rates))
        self.measures()  # Fail early for invalid transitions or output-name collisions.

    def measures(self) -> tuple[Sum | Count | Ratio, ...]:
        """Build quantity measures followed by rates, without fitting or computing."""
        names = [s.name for s in self.stages]
        quantities = [replace(s.measure, name=f"{s.name}数量") for s in self.stages]
        if isinstance(self.rates, str):
            pairs = []
            if self.rates in ("adjacent", "both"):
                pairs.extend(zip(names, names[1:]))
            if self.rates in ("from_first", "both"):
                pairs.extend((names[0], name) for name in names[1:])
            transitions = [Transition(a, b) for a, b in dict.fromkeys(pairs)]
        else:
            transitions = self.rates
        seen = set()
        result = list(quantities)
        for transition in transitions:
            if not isinstance(transition, Transition):
                raise MeasureError("Explicit rates must contain Transition objects")
            before, after = transition.before, transition.after
            if before not in names or after not in names:
                raise MeasureError(f"Unknown stage in transition {before!r} -> {after!r}")
            if names.index(before) >= names.index(after):
                raise MeasureError("Transitions must go forward between distinct stages")
            if (before, after) in seen:
                raise MeasureError("Duplicate transition")
            seen.add((before, after))
            result.append(
                Ratio(
                    numerator=f"{after}数量",
                    denominator=f"{before}数量",
                    name=transition.name or f"{before}→{after}转化率",
                )
            )
        if len({m.name for m in result}) != len(result):
            raise MeasureError("Funnel output measure names must be unique")
        return tuple(result)
