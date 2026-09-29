"""A method space: native-code recording and queries over its own store."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .record import Run
from .record.query import select_records
from .record.session import RunSession
from .record.store import ExperimentStore

if TYPE_CHECKING:
    import pandas as pd

    from .configuration import ConfigSource


class MethodExperiment:
    def __init__(self, store: ExperimentStore):
        self._store = store
        self._identity = dict(store.method)

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def config_dir(self) -> Path:
        return self.path / "configs"

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    @property
    def comparison_partitions(self) -> tuple[str, ...] | None:
        return self._store.comparison_partitions

    def start_run(
        self,
        *,
        name: str = "",
        params: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        comparison_partitions: Iterable[str] | None = None,
        config: "Mapping[str, Any] | str | Path | ConfigSource | None" = None,
        task: Mapping[str, Any] | None = None,
    ) -> RunSession:
        """Snapshot inputs, then record native code; never train a model here.

        task optionally supplies objective/metric/feval display identities for any backend.
        """
        return RunSession(
            self._store,
            name=name,
            config=config,
            params=params,
            metadata=metadata,
            comparison_partitions=(
                self.comparison_partitions
                if comparison_partitions is None
                else comparison_partitions
            ),
            task=task,
            model={
                "backend": self._identity["name"],
                "family": self._identity["family"],
                "backend_version": "unrecorded",
            },
        )

    def get_run(self, reference: str, *, verify: bool = True) -> Run:
        record = select_records(self._store, [reference])[0]
        return Run.load(self._store.run_path(record.run_id), verify=verify)

    def runs(self, *, status: str | None = None) -> "pd.DataFrame":
        from .record.comparison import index_records

        return index_records(self._store.list_records(), status=status)

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
        from .record.comparison import compare_records

        return compare_records(
            select_records(self._store, runs),
            metrics=metrics,
            partitions=partitions,
            default_partitions=self.comparison_partitions,
            params=params,
            features=features,
            metadata=metadata,
            fields=fields,
        )

    def compare_params(
        self, left: str, right: str, *, params=None, only_changed: bool = True
    ) -> "pd.DataFrame":
        from .record.comparison import compare_parameters

        return compare_parameters(
            *select_records(self._store, [left, right]), params=params, only_changed=only_changed
        )
