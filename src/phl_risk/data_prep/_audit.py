from dataclasses import dataclass

from phl_risk._data import freeze

from ._snapshot import PrepSnapshot


@dataclass(frozen=True)
class PrepAudit:
    step: str
    input_columns: tuple
    output_columns: tuple
    before: PrepSnapshot
    after: PrepSnapshot
    details: object
    examples: tuple = ()
    kind: str | None = None

    def __post_init__(self):
        for key in ("input_columns", "output_columns", "details", "examples"):
            object.__setattr__(self, key, freeze(getattr(self, key)))
