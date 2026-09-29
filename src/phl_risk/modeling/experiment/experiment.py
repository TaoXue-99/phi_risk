"""Coordinate independent prepared-data attempts and query their recorded facts."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .record import Run
from .record.query import select_records
from .record.store import ExperimentStore

if TYPE_CHECKING:
    import pandas as pd

    from .configuration import ConfigSource
    from .method import MethodExperiment
    from .record.session import RunSession


class Experiment:
    """A collection of immutable attempts. Opening an experiment needs no input data."""

    def __init__(self, *, root: str | Path, name: str | None = None):
        """Use root directly; an explicit name retains the legacy root/name layout."""
        direct = name is None
        name = name if name is not None else Path(root).expanduser().resolve().name
        self._store = ExperimentStore(Path(root), name, direct=direct)
        self.name = name

    @property
    def path(self) -> Path:
        return self._store.path

    def get_run(self, reference: str, *, verify: bool = True) -> Run:
        record = select_records(self._store, [reference])[0]
        return Run.load(self._store.run_path(record.run_id), verify=verify)

    def runs(self, *, status: str | None = None) -> "pd.DataFrame":
        """Return a lightweight fact index, including failed and running attempts."""
        from .record.comparison import index_records

        return index_records(self._store.list_records(), status=status)

    def start_run(
        self,
        *,
        name: str = "run",
        params: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        comparison_partitions: Iterable[str] | None = None,
        model: Mapping[str, Any] | None = None,
        config: "Mapping[str, Any] | str | Path | ConfigSource | None" = None,
        task: Mapping[str, Any] | None = None,
    ) -> "RunSession":
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
            task=task,
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
    ) -> "pd.DataFrame":
        """Select and flatten completed records; no automatic gap or delta calculations."""
        from .record.comparison import compare_records

        return compare_records(
            select_records(self._store, runs),
            metrics=metrics,
            partitions=partitions,
            params=params,
            features=features,
            metadata=metadata,
            fields=fields,
        )

    def initialize(
        self,
        *,
        method: str,
        family: str | None = None,
        objective=None,
        metric=None,
        comparison_partitions: Iterable[str] | None = None,
    ) -> "MethodExperiment":
        from .initialization import initialize

        return initialize(
            self,
            method,
            family=family,
            objective=objective,
            metric=metric,
            comparison_partitions=comparison_partitions,
        )

    def open_method(self, method: str) -> "MethodExperiment":
        from .initialization import open_method

        return open_method(self, method)

    def compare_params(self, left, right, *, params=None, only_changed=True):
        from .record.comparison import compare_parameters

        records = select_records(self._store, [left, right])
        return compare_parameters(*records, params=params, only_changed=only_changed)
