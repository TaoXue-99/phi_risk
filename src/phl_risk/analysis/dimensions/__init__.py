"""Analytical dimensions, distinct from result axes."""

from ._base import BaseDimension
from ._bin import BinDimension
from ._column import ColumnDimension

DimensionLike = str | BaseDimension

__all__ = ["BaseDimension", "ColumnDimension", "BinDimension", "DimensionLike"]
