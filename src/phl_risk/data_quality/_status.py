from enum import Enum


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


def worst(statuses):
    ranks = {
        s: i
        for i, s in enumerate(
            (CheckStatus.SKIP, CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL)
        )
    }
    return max(statuses, key=ranks.__getitem__, default=CheckStatus.SKIP)
