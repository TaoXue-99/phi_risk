# LightGBM Experiment implementation plan

Goal: implement the user-specified binary LightGBM execution layer without changing Plan APIs.
Architecture: immutable Run snapshots, atomic filesystem Store, native Dataset/trainer, shared AUC,
reference comparison, optional Hydra composition, sklearn RFE candidates retrained natively.
Spec: the 63-section task supplied in this conversation.

Global constraints: Python >=3.12; LightGBM >=4.0,<5 optional; Hydra optional; no AutoML;
no raw data or predictions persisted; numerical DataFrame metrics; declaration imports backend-free.

- [x] Phase 1: tests for runtime contracts, deterministic ordered SHA256 splits, config, store/reload.
  Implement config.py, split.py, runtime.py, store.py, run.py, _utils.py and exceptions.
- [x] Phase 2: test weighted native training, schema, history, failure artifacts and lazy loading.
  Implement dataset.py, callbacks.py, trainer.py, evaluation.py and experiment.py.
- [x] Phase 3: test reference ambiguity, feature membership/count differences, recursive parameter
  differences, numeric deltas and OOT hiding. Implement comparison.py.
- [x] Phase 4: test real composition, interpolation, overrides and absent dependencies.
  Implement hydra.py, exports and optional dependency groups.
- [x] Phase 5: test train-only RFE, exact feature counts and native candidate retraining.
  Implement feature_selection.py and Experiment.select_features.
- [x] Phase 6: runnable example, baseline YAML, README and execution documentation;
  focused/full pytest, ruff check/format, example execution, final diff review.

Each phase starts with failing behavioral tests, then implementation and focused verification.
Execution is inline in the current repository as requested. No Plan redesign, automatic model
selection, publishing or remote changes are part of this task.
