"""A thin method-scoped facade; execution and records use the ordinary Experiment chain."""

from pathlib import Path

from .experiment import Experiment


class MethodExperiment:
    def __init__(self, store):
        self._session = Experiment._from_store(store)
        self._identity = dict(store.method)

    @property
    def path(self) -> Path:
        return self._session.path

    @property
    def config_dir(self) -> Path:
        return self.path / "configs"

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    def start_run(
        self, *, name="", params=None, metadata=None, comparison_partitions=None, config=None
    ):
        return self._session.start_run(
            name=name,
            config=config,
            params=params,
            metadata=metadata,
            comparison_partitions=comparison_partitions,
            model={
                "backend": self._identity["name"],
                "family": self._identity["family"],
                "backend_version": "unrecorded",
            },
        )

    def get_run(self, reference):
        return self._session.get_run(reference)

    def runs(self, *, status=None):
        return self._session.runs(status=status)

    def compare(self, **options):
        return self._session.compare(**options)

    def compare_params(self, left, right, *, params=None, only_changed=True):
        return self._session.compare_params(left, right, params=params, only_changed=only_changed)
