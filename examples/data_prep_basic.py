import pandas as pd
from sklearn.preprocessing import RobustScaler

from phl_risk.data_prep import DataPrep
from phl_risk.data_prep.steps import MissingImputer, SklearnStep, ToNumeric

train = pd.DataFrame({"income": ["100", "200", None]})
oot = pd.DataFrame({"income": ["unknown", "10000"]})
prep = DataPrep(
    [
        ToNumeric(["income"], "coerce"),
        MissingImputer(["income"]),
        SklearnStep(RobustScaler(), ["income"]),
    ]
).fit(train)
result = prep.run(oot)
assert len(result.audits) == 3
print(result.data)
print(result.audit_frame())
