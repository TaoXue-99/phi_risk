"""Internal calculation-time evidence, separate from canonical metric values."""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Diagnostic:
    """One cause for a canonical row; several causes may share a row."""

    row: int
    reason: str
    message: str
    field: str | None = None
    input_rows: int | None = None
    affected_rows: int | None = None
    dependency: str | None = None


def diagnostic_table(
    data: pd.DataFrame, axis_names: list[str], records: tuple[Diagnostic, ...]
) -> pd.DataFrame:
    """Return observed nulls only; never infer a cause from the null itself."""
    by_row: dict[int, list[Diagnostic]] = {}
    for record in records:
        by_row.setdefault(record.row, []).append(record)
    coordinates, entries = [], []
    for position in data["value"].isna().to_numpy().nonzero()[0]:
        causes = by_row.get(int(position)) or [
            Diagnostic(
                int(position),
                "reason_not_recorded",
                "This calculation has not recorded a detailed cause.",
            )
        ]
        for cause in causes:
            coordinates.append(tuple(data.iloc[position][name] for name in axis_names))
            entry = asdict(cause)
            del entry["row"]
            entries.append(entry)
    index = (
        pd.MultiIndex.from_tuples(coordinates, names=axis_names)
        if len(axis_names) > 1
        else pd.Index([key[0] for key in coordinates], name=axis_names[0])
    )
    table = pd.DataFrame(
        entries,
        index=index,
        columns=["reason", "field", "input_rows", "affected_rows", "dependency", "message"],
    )
    for column in ("input_rows", "affected_rows"):
        table[column] = pd.array(table[column], dtype="Int64")
    return table
