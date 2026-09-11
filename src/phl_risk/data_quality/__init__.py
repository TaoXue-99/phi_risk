"""Reference-based data quality checks and immutable reports."""

from ._base import BaseQualityCheck
from ._profile import ColumnProfile, DataProfile
from ._quality import DataQuality
from ._result import CheckResult, QualityReport
from ._status import CheckStatus
from .checks import (
    CardinalityCheck,
    CategorySetCheck,
    ConstantCheck,
    FiniteCheck,
    MissingLikeCheck,
    MissingRateCheck,
    NumericConvertibleCheck,
    RangeCheck,
    RowCountCheck,
    SchemaCheck,
    UniqueCheck,
)

__all__ = [
    "BaseQualityCheck",
    "ColumnProfile",
    "DataProfile",
    "DataQuality",
    "CheckResult",
    "QualityReport",
    "CheckStatus",
    "CardinalityCheck",
    "CategorySetCheck",
    "ConstantCheck",
    "FiniteCheck",
    "MissingLikeCheck",
    "MissingRateCheck",
    "NumericConvertibleCheck",
    "RangeCheck",
    "RowCountCheck",
    "SchemaCheck",
    "UniqueCheck",
]
