from dataclasses import FrozenInstanceError

import joblib
import pandas as pd
import pytest
from sklearn.base import clone

from phl_risk.data_quality import (
    BaseQualityCheck,
    CheckResult,
    CheckStatus,
    DataQuality,
    QualityReport,
)
from phl_risk.exceptions import NotFittedError


class DummyStaticCheck(BaseQualityCheck):
    requires_fit = False

    def _validate(self, X):
        X.iloc[0, 0] = 0
        return CheckResult("static", CheckStatus.FAIL, ("x",), "test", details={"a": [1]})


class DummyReferenceCheck(BaseQualityCheck):
    def _fit(self, X, y=None):
        self.rows_ = len(X)

    def _validate(self, X):
        return CheckResult("reference", CheckStatus.PASS, tuple(X.columns), "ok", len(X))


def test_lifecycle_clone_freeze_and_all_checks():
    X = pd.DataFrame({"x": [1, 2, None], "s": ["a", "b", "a"]})
    original = X.copy()
    static, ref = DummyStaticCheck(), DummyReferenceCheck()
    assert static.validate(X).failed
    with pytest.raises(NotFittedError):
        ref.validate(X)
    quality = DataQuality([static, ref]).fit(X)
    assert not hasattr(ref, "rows_")
    assert not hasattr(clone(quality), "checks_")
    before = joblib.hash(quality)
    report = quality.validate(X)
    assert len(report.results) == 2 and report.failed
    assert joblib.hash(quality) == before
    pd.testing.assert_frame_equal(original, X)
    assert quality.reference_profile_.profiles["x"].missing_count == 1
    assert quality.reference_profile_.profiles["s"].top_values == (("a", 2), ("b", 1))
    with pytest.raises(FrozenInstanceError):
        report.results = ()
    with pytest.raises(TypeError):
        report.results[0].details["a"] = 2
    assert QualityReport(()).status == CheckStatus.SKIP
    assert list(report.to_frame()) == [
        "check",
        "status",
        "columns",
        "observed",
        "expected",
        "message",
    ]
    quality.set_params(checks=[static])
    with pytest.raises(NotFittedError):
        quality.validate(X)
