import pandas as pd

from phl_risk.data_quality.checks import CardinalityCheck


def test_incident():
    check = CardinalityCheck(["x"], fail_change=0.5).fit(pd.DataFrame({"x": range(100)}))
    assert check.validate(pd.DataFrame({"x": [1] * 100})).failed
