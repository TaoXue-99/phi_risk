import pandas as pd

from phl_risk.data_quality.checks import SchemaCheck


def test_incident():
    ref = pd.DataFrame({f"x{i}": [1, 2] for i in range(10)})
    check = SchemaCheck().fit(ref)
    r = check.validate(ref.drop(columns=["x8", "x9"]))
    assert r.failed and r.details["missing_columns"] == ("x8", "x9")
    assert check.validate(pd.concat([ref, ref[["x0"]]], axis=1)).failed
    assert check.validate(ref[ref.columns[::-1]]).failed
