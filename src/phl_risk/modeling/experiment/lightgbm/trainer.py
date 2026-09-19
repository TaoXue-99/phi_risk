"""The native training boundary; no persistence, comparisons or feature selection."""

from dataclasses import dataclass
from time import perf_counter

from ._utils import require
from .callbacks import build_callbacks
from .dataset import DatasetPair


@dataclass(frozen=True)
class LightGBMFitResult:
    booster: object
    best_iteration: int
    best_score: dict
    eval_history: dict
    training_seconds: float


class LightGBMTrainer:
    def fit(self, datasets: DatasetPair, config: dict) -> LightGBMFitResult:
        lgb = require("lightgbm", "lightgbm")
        history = {}
        callbacks = build_callbacks(config["train"], history)
        start = perf_counter()
        booster = lgb.train(
            params=config["model"]["params"],
            train_set=datasets.train,
            num_boost_round=config["train"]["num_boost_round"],
            valid_sets=[datasets.train, datasets.validation],
            valid_names=["train", datasets.validation_name],
            callbacks=callbacks,
        )
        seconds = perf_counter() - start
        iteration = (
            booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()
        )
        return LightGBMFitResult(booster, iteration, dict(booster.best_score), history, seconds)
