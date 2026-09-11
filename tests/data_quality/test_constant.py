import pandas as pd

from phl_risk.data_quality.checks import ConstantCheck


def test_incident():
    assert ConstantCheck().validate(pd.DataFrame({"x": [0.5] * 100})).failed
    assert (
        ConstantCheck(max_dominant_rate=0.99).validate(pd.DataFrame({"x": [0] * 999 + [1]})).failed
    )
