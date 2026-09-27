"""Coordinate independent prepared-data attempts and query their recorded facts."""

from collections.abc import Sequence
from pathlib import Path

from phl_risk.exceptions import ExperimentError

from .record import Run
from .record.store import ExperimentStore


class Experiment:
    """A collection of immutable attempts. Opening an experiment needs no input data."""

    def __init__(self, *, root: str | Path, name: str | None = None):
        """Use root directly; an explicit name retains the legacy root/name layout."""
        direct = name is None
        name = name if name is not None else Path(root).expanduser().resolve().name
        self._store = ExperimentStore(Path(root), name, direct=direct)
        self.name = name

    @classmethod
    def _from_store(cls, store):
        """Bind an existing store without creating or modifying its manifest."""
        session = cls.__new__(cls)
        session._store = store
        session.name = store.path.name
        return session

    @property
    def path(self) -> Path:
        return self._store.path

    def _select_records(self, runs=None):
        records = self._store.list_records()
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

    def get_run(self, reference: str) -> Run:
        record = self._select_records([reference])[0]
        return Run.load(self._store.run_path(record.run_id))

    def runs(self, *, status: str | None = None):
        """Return a lightweight fact index, including failed and running attempts."""
        from .record.comparison import index_records

        return index_records(self._store.list_records(), status=status)

    def start_run(
        self,
        *,
        name="run",
        params=None,
        metadata=None,
        comparison_partitions=None,
        model=None,
        config=None,
    ):
        """Record a block of ordinary Python code; never invoke a model trainer."""
        from .record.session import RunSession

        return RunSession(
            self._store,
            name=name,
            params=params,
            metadata=metadata,
            comparison_partitions=comparison_partitions,
            model=model,
            config=config,
        )

    def compare(
        self,
        *,
        runs=None,
        metrics=None,
        partitions=None,
        params=None,
        features: bool = False,
        metadata=None,
        fields=None,
    ):
        """Select and flatten completed records; no automatic gap or delta calculations."""
        from .record.comparison import compare_records

        return compare_records(
            self._select_records(runs),
            metrics=metrics,
            partitions=partitions,
            params=params,
            features=features,
            metadata=metadata,
            fields=fields,
        )

    def initialize(self, *, method, family=None, objective=None, metric=None):
        from .initialization import initialize

        return initialize(self, method, family=family, objective=objective, metric=metric)

    def open_method(self, method):
        from .initialization import open_method

        return open_method(self, method)

    def compare_params(self, left, right, *, params=None, only_changed=True):
        from .record.comparison import compare_parameters

        records = self._select_records([left, right])
        return compare_parameters(*records, params=params, only_changed=only_changed)
