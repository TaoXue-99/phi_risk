import pandas as pd

from phl_risk.data_quality.checks import FiniteCheck


def test_incident():
    X = pd.DataFrame({"x": [1.0, float("inf"), float("-inf"), None]})
    r = FiniteCheck(["x"]).validate(X)
    assert r.failed and r.details["x"]["positive_inf"] == 1
    assert FiniteCheck(["x"]).validate(X.iloc[[0, 3]]).passed
