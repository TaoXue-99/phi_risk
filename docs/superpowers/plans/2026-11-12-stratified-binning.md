# Stratified Binning Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add opt-in per-stratum bin fitting and group-only totals, with a runnable Chinese guide.

**Architecture:** BinDimension owns independent fitted transformers keyed by raw fit_by values. Plans validate raw grouping dimensions, engines retain stratification axes for totals, and layout resolves intervals with their stratum context through a dedicated renderer.

**Tech Stack:** Python, pandas, NumPy, pytest, nbclient.

**Spec:** docs/superpowers/specs/2026-11-12-stratified-binning-design.md

## Global Constraints

- fit_by=None preserves existing global behavior.
- Never refit during compute; unseen strata raise.
- Totals retain every fit_by field; no global total or cross-stratum bin total.
- Missing grouping keys follow the existing drop/keep dimension policy.

## Review Focus

- Duplicate input indices must preserve positional assignment.
- Custom labels and different actual bin counts must not invent intervals.
- Empty current data must render without refitting.
- Independent stratified bin dimensions must retain all grouping keys in totals.
- Splitting grouping and bin axes across rows/columns requires a sparse interval union, never a false shared interval.

## Task 1: Per-stratum fitting

Files: dimensions/_bin.py, dimensions/_stratified.py, _cube.py, _plan.py, engines/_pandas.py, engines/_comparison.py; tests/test_stratified_binning.py.
Interfaces: BinDimension(..., fit_by=None), metadata fit_by + groups; grouping keys use raw fields with unrenamed ColumnDimension axes.
- [x] Add tests comparing stratified results with independently fitted cubes, reference reuse, missing/drop/keep, duplicate indices, multi-key strata, failed refit, invalid declarations and custom labels.
- [x] Run tests and observe new API failures.
- [x] Implement independent atomic fit state and positional transforms; supply missing policy through fit/compute preparation.
- [x] Run focused tests, retain global regression tests.

## Task 2: Totals and context-sensitive layout

Files: _plan.py, engines/_pandas.py, _result.py, _layout.py, _stratified_layout.py; tests/test_stratified_binning.py.
Interfaces: protected total dimensions in metadata; grouped transformer metadata contains keys and transform metadata.
- [x] Add tests for group totals, prohibited collapse, code/interval row/column layouts, varying bin counts, two stratified dimensions, empty data and max_cells.
- [x] Run tests and observe failures.
- [x] Filter grouping sets at computation; protect total API; render contextual intervals using valid coordinate domains and a union when axes are split.
- [x] Run focused tests and all analysis tests.

## Task 3: Documentation and acceptance

Files: examples/analysis_complete_guide.ipynb, docs/analysis_binning.md.
- [x] Add self-contained deterministic examples explaining global score intervals versus within-group ranks, fit/compute reuse, multi-field grouping and group-only totals.
- [x] Execute full notebook, retaining inspected outputs.
- [x] Run complete pytest suite, ruff on changed code and git diff --check; review the final changes.

## Execution record

- Baseline: 51 existing cube/transform/binner tests passed.
- Work in the existing managed checkout on branch work; no main branch mutation or remote publishing is needed.
- User approved the design and implementation; continue without another plan approval step.

- Tasks 1–3 implemented. New API initially failed 14 tests; implementation passed. Additional lifecycle/layout regressions were tested RED→GREEN.
- Independent review found duplicate-index supervised filtering rejected valid input. Fixed by preserving original integer row positions through filtering and missing-stratum selection; four supervised combinations are covered.
- Final self-review also verified requested axis ordering and generic backend data compatibility with dedicated regression tests.
- Notebook executed from a fresh kernel: 40 code cells, no errors. The plotting dependency was installed only into the local virtual environment; project dependencies were not changed.
