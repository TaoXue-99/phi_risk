import pandas as pd

from phl_risk.data_quality.checks import CategorySetCheck


def test_incident():
    check = CategorySetCheck("x").fit(pd.DataFrame({"x": ["a", "b", "c"]}))
    r = check.validate(pd.DataFrame({"x": ["a", "b", "d"]}))
    assert r.failed
    assert r.details["new_categories"] == ("d",)
    assert r.details["missing_categories"] == ("c",)
