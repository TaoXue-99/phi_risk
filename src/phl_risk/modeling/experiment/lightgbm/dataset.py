"""Build fresh native datasets for every attempt."""

from dataclasses import dataclass

from ._utils import require
from .runtime import RuntimeData


@dataclass(frozen=True)
class DatasetPair:
    train: object
    validation: object
    validation_name: str


class LGBDatasetBuilder:
    def build(
        self, runtime: RuntimeData, features: tuple[str, ...], validation: str
    ) -> DatasetPair:
        lgb = require("lightgbm", "lightgbm")
        train, valid = runtime.partitions["train"], runtime.partitions[validation]
        categorical = [f for f in features if f in runtime.categorical]
        train_set = lgb.Dataset(
            train[list(features)],
            label=train[runtime.target],
            weight=None if runtime.weight is None else train[runtime.weight],
            feature_name=list(features),
            categorical_feature=categorical,
        )
        valid_set = train_set.create_valid(
            valid[list(features)],
            label=valid[runtime.target],
            weight=None if runtime.weight is None else valid[runtime.weight],
        )
        return DatasetPair(train_set, valid_set, validation)
