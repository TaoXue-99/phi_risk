"""Independent declaration and execution entry points, imported only on demand."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plan import DataPlan, ModelPlan

__all__ = ["ModelPlan", "DataPlan"]


def __getattr__(name):
    if name in __all__:
        from . import plan

        return getattr(plan, name)
    raise AttributeError(name)
