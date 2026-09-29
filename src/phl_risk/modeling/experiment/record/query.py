"""Shared record selection for project and method spaces; no facade dependencies."""

from collections.abc import Sequence

from phl_risk.exceptions import ExperimentError


def select_records(store, runs=None):
    records = store.list_records()
    if runs is None:
        return records
    if isinstance(runs, str):
        runs = [runs]
    if not isinstance(runs, Sequence) or any(not isinstance(r, str) for r in runs):
        raise ExperimentError("runs must be a sequence of IDs or names")
    selected = []
    for reference in runs:
        exact = [r for r in records if r.run_id == reference]
        matches = exact or [r for r in records if r.to_dict().get("run") == reference]
        matches = matches or [r for r in records if r.name == reference]
        if len(matches) != 1:
            raise ExperimentError(
                f"Run reference {reference!r}: "
                f"{'ambiguous; use run_id' if matches else 'not found'}"
            )
        if matches[0].run_id in [r.run_id for r in selected]:
            raise ExperimentError("Duplicate Run selection")
        selected.append(matches[0])
    return tuple(selected)
