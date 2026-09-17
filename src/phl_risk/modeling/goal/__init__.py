"""Column-independent learning goals."""

from ._base import ModelingGoal
from .causal import CausalEffect
from .classification import BinaryClassification
from .regression import Regression
from .survival import Survival

__all__ = ["ModelingGoal", "BinaryClassification", "Regression", "CausalEffect", "Survival"]
