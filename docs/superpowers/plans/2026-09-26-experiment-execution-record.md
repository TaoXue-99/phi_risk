# Experiment Execution / Record implementation plan

User approved the corrected architecture and implementation, including example updates.
This supersedes the backend-local structure in the earlier 0.6 draft. Execute inline.

## Boundaries

- Experiment orchestrates prepared execution and recording, never imports Tree/Deep backends.
- execution/result.py defines PreparedExecution and ExecutionResult: fact mappings, execution callable, artifact writers.
- execution/tree/lightgbm owns input/config validation, training, AUC, model serialization/loading and RFE.
- execution/deep is a documented namespace only, with no invented deep trainer.
- record owns RunRecord/Run, artifact descriptors, integrity, Store and generic comparison; no backend or config resolver imports.
- adapters/hydra.py remains optional and composes LightGBM configuration only when called.
- Record schema version becomes 3 so interim backend-bound v2 is not silently misread.
- Package remains unpublished 0.6.0. No commits/pushes or unrelated Plan edits.

## Test-first tasks

- [x] Core record tests: backend import blockade; non-tree records without AUC/best_iteration; arbitrary artifacts; stable state and integrity.
- [x] Move generic record/store/comparison; validate common fields without forcing LightGBM data/config conventions.
- [x] Add execution result contract and top-level Experiment orchestration, including preparation failure and execution failure.
- [x] Move backend implementation into execution/tree/lightgbm; split model serialization out of Store, keep native training and 4.0 compatibility.
- [x] Migrate integration tests to Experiment + LightGBMExecution, explicit load_model and RFE candidate helper.
- [x] Update executable example, README, use/migration docs and review evidence.
- [x] Run full pytest/Ruff, current and 4.0 backend tests, base installation, examples and wheel checks.

## Review focus

Generic records must not require target/train/valid/AUC/Booster params. Model loading must validate backend and schema only when explicitly requested. Artifact paths cannot replace run.json or escape Run. Backend callbacks cannot mutate recorded initial facts. Generic comparison must handle different metric/config structures and preserve numerical values.
