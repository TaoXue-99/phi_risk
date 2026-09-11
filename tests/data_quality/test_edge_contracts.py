import joblib
import pandas as pd
import pytest
from sklearn.base import clone

from phl_risk.data_quality import (
    CategorySetCheck,
    CheckStatus,
    ConstantCheck,
    DataProfile,
    DataQuality,
    FiniteCheck,
    MissingLikeCheck,
    MissingRateCheck,
    NumericConvertibleCheck,
    RangeCheck,
    RowCountCheck,
    SchemaCheck,
    UniqueCheck,
)
from phl_risk.exceptions import DataQualityError


def test_profile_nullable_boolean_bounded_top_values():
    X = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None], dtype="boolean"),
            "n": pd.Series([1, None, 3], dtype="Int64"),
            "s": ["a", "b", "c"],
        }
    )
    profile = DataProfile.from_frame(X, max_top_values=1)
    assert profile.profiles["flag"].quantiles["q50"] == 0.5
    assert profile.profiles["n"].missing_count == 1
    assert len(profile.profiles["s"].top_values) == 1


@pytest.mark.parametrize(
    "check",
    [
        MissingLikeCheck(["x"]),
        UniqueCheck(["x"]),
        ConstantCheck(["x"]),
        NumericConvertibleCheck(["x"]),
        FiniteCheck(["x"]),
        RangeCheck("x", 0, 1),
    ],
)
def test_empty_current_is_skip(check):
    assert check.validate(pd.DataFrame({"x": pd.Series(dtype=float)})).status == CheckStatus.SKIP


def test_schema_controls_and_atomic_quality_refit():
    X = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert SchemaCheck(allow_extra=True).fit(X).validate(X.assign(z=1)).passed
    assert SchemaCheck(check_order=False).fit(X).validate(X[["y", "x"]]).passed
    assert SchemaCheck().fit(X).validate(X.astype(float)).failed
    quality = DataQuality([MissingRateCheck(["x"])]).fit(X)
    before = joblib.hash(quality)
    with pytest.raises(DataQualityError):
        quality.fit(X.drop(columns="x"))
    assert joblib.hash(quality) == before
    assert not hasattr(clone(quality), "checks_")


def test_zero_reference_rows():
    check = RowCountCheck(fail_relative_change=0.1).fit(pd.DataFrame({"x": []}))
    assert check.validate(pd.DataFrame({"x": [1]})).failed


def test_category_switches_record_both_issues():
    check = CategorySetCheck("x", check_new=False, check_missing=False).fit(
        pd.DataFrame({"x": ["a", "b"]})
    )
    report = check.validate(pd.DataFrame({"x": ["c"]}))
    assert report.passed and report.details["new_categories"] == ("c",)
    assert report.details["missing_categories"] == ("a", "b")
