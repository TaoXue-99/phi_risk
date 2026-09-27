# LightGBM Experiment 0.6 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. User explicitly authorized implementation after approving the design.

**Goal:** Independent prepared-data experiments that save facts and expose queryable immutable records.
**Architecture:** Preserve native execution and atomic filesystem artifacts; replace Plan-coupled orchestration with explicit per-run input, immutable config and schema-v2 records.
**Tech Stack:** Python 3.12+, pandas, LightGBM >=4.0,<5, sklearn RFE, pytest, Ruff.
**Spec:** ../specs/2026-09-26-lightgbm-experiment-v06-design.md

## Global constraints

No Plan/Split dependency, data transformations, parent relations, automatic winner or persisted derived metrics. Preserve ongoing unrelated user edits. No automatic commits/pushes. Hydra optional. Old artifacts rejected without mutation.

## Review focus

- Concurrent allocation and partially written artifacts: Store tests for unique IDs and interrupted completion.
- Nested mutable config/record values: defensive-copy and mutation tests.
- Artifact path traversal and inconsistent manifest/model: loader tests.
- Category schema mismatch and effective class weights: input tests.
- Cross-partition and parameter column naming collisions: comparison tests.

## Tasks

### 1. Record foundation
- [x] Write failing tests in test_records.py/test_store.py for ExperimentStore(root,name), allocate(name,facts), fail, list_records, RunRecord and LightGBMRun.load.
- [x] Rewrite run.py/store.py: version=2, all statuses, timestamps, artifact descriptors, atomic writes; no persisted contract/reference. Keep read-only legacy-version rejection.
- [x] Run targeted pytest; check failure leaves no invented result.

### 2. Config
- [x] Write test_config.py: defaults, supplied/resolved, with_params/with_overrides, alias changes, deep immutability, controls.
- [x] Add LightGBMConfig and backend-independent ResolvedConfig in config.py; retain validator; update hydra.py adapter.
- [x] Run config/Hydra tests.

### 3. Execution
- [x] Rewrite fixtures/tests for exp(name,root).run(partitions,target,features,config,...); test input failures, category contracts, identity, models and failed lifecycle.
- [x] Replace runtime.py by input.py validate_input(...) returning ExperimentInput; delete split.py; retain fresh Dataset builder, native Trainer/callbacks; extract importance.py.
- [x] Orchestrate start/train/evaluate/complete and preserve original exception; nested AUC only.
- [x] Run execution tests including weighted model roundtrip, absent optional backend, missing validation, arbitrary partition.

### 4. Comparison
- [x] Tests for runs(), filtering, nested metric flatten, params/features/metadata selection, no forced gap/delta, column conflicts.
- [x] Rewrite comparison.py with facts-only index and compare_records; load manifests only for queries.
- [x] Run comparison/store tests.

### 5. RFE
- [x] Tests proving train-only, weights, category handling, native candidates and no private Experiment state.
- [x] Pure select_candidates(input,config,counts,step,iterations) produces feature tuples; Experiment explicitly runs each. Remove tolerance/relation state.
- [x] Run RFE and backend compatibility tests.

### 6. Integration and documentation
- [x] Update example to prepared partitions and immutable Python config; optional Hydra separately.
- [x] Update README diagrams/entry points, migration docs, CHANGELOG, version 0.6.0 and uv.lock; remove stale Experiment claims in Plan tutorial only.
- [x] Run full pytest -W error, Ruff check/format, example, isolated LightGBM 4.0 and current backend; report any failures/limits.

## Execution ledger

- Record foundation: running/failed snapshots and legacy rejection RED → GREEN; Store interruption/concurrency tests added.
- Config: missing LightGBMConfig RED → supplied/resolved/immutable tests GREEN; Hydra adapter retained.
- Execution: old constructor rejection RED → prepared partitions/native model tests GREEN. Runtime/split files deleted.
- Comparison: missing index/compare functions RED → facts-only filtered views GREEN; no model I/O in index.
- RFE: missing explicit-input helper RED → train-only weighted/native candidate tests GREEN.
- Review fixes: import blockade revealed parent modeling eager import; changed only its public forwarding to lazy imports, preserving Plan class identity.
- Review fixes: partial DART override, conflicting regularizer aliases, eta/max_leaves shadowing, missing Hydra provenance and malformed completed records all have regression tests.
- Ruling: no compatibility facade for previous APIs/artifacts, as user approved a new iteration. Legacy directories are rejected without rewriting.
- Ruling: JSON manifest is the single config/metrics/features source; PyYAML removed from core optional dependencies. Hydra uses its own YAML dependency.
- Ruling: aliases needed by native-to-sklearn RFE reuse the configuration alias groups to prevent wrapper defaults shadowing user values.
- Final local full environment: 493 passed; LightGBM 4.0.0: 77 passed; Ruff lint/format passed. See review document for complete environment/packaging evidence.
