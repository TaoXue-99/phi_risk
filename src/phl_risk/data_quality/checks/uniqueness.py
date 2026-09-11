from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import examples_limit, result, selected


class UniqueCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(self, columns, max_examples=10):
        self.columns = columns
        self.max_examples = max_examples

    def _validate(self, X):
        examples_limit(self.max_examples)
        columns = selected(X, self.columns)
        mask = X.duplicated(list(columns), keep=False)
        details = dict(
            duplicate_count=int(mask.sum()),
            duplicate_rate=float(mask.mean()),
            examples=X.loc[mask, list(columns)].head(self.max_examples).to_dict("records"),
        )
        status = CheckStatus.FAIL if mask.any() else CheckStatus.PASS
        return result(self, columns, details, [status if len(X) else CheckStatus.SKIP])
