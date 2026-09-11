import pandas as pd
import pytest

from phl_risk.data_quality.checks import MissingRateCheck


def test_incident():
    check = MissingRateCheck(["x"], warn_delta=0.1, fail_delta=0.2)
    check.fit(pd.DataFrame({"x": [None] * 2 + [1] * 98}))
    r = check.validate(pd.DataFrame({"x": [None] * 35 + [1] * 65}))
    assert r.failed and r.details["x"]["delta"] == pytest.approx(0.33)
