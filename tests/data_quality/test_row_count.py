import pandas as pd

from phl_risk.data_quality import CheckStatus
from phl_risk.data_quality.checks import RowCountCheck


def test_incident():
    check = RowCountCheck(min_rows=2, warn_relative_change=0.1, fail_relative_change=0.5)
    check.fit(pd.DataFrame({"x": range(10)}))
    assert check.validate(pd.DataFrame({"x": range(8)})).status == CheckStatus.WARN
    assert check.validate(pd.DataFrame({"x": [1]})).failed
