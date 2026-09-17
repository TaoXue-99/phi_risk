"""CausalEffect learning semantics."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelingGoal


@dataclass(frozen=True)
class CausalEffect(ModelingGoal):
    """Declare causal semantics without binding real data columns."""

    name: ClassVar[str] = "CausalEffect"
    family: ClassVar[str] = "causal"
    required_roles: ClassVar[frozenset[str]] = frozenset(("outcome", "treatment"))
    objective_families: ClassVar[tuple[str, ...]] = ("causal_effect",)
