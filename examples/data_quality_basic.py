"""Run with: uv run python examples/data_quality_basic.py."""

import pandas as pd

from phl_risk.data_quality import DataQuality
from phl_risk.data_quality.checks import MissingRateCheck, SchemaCheck

train = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
oot = pd.DataFrame({"income": [100.0, None, None]})
quality = DataQuality([SchemaCheck(), MissingRateCheck(["income"], fail_delta=0.2)]).fit(train)
report = quality.validate(oot)
assert report.failed
print(report.to_frame())
