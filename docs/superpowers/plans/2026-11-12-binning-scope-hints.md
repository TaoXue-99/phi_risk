# Binning Scope Hints Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Clarify global versus within-group bin fitting without warnings or result changes.
**Architecture:** An internal scope helper supplies INFO hint text and explain descriptions; Cube emits hints after successful fitting, explain remains read-only.
**Tech Stack:** Python logging, pandas, pytest caplog, Jupyter nbclient.
**Spec:** docs/superpowers/specs/2026-11-12-binning-scope-hints-design.md

## Global Constraints

English runtime text; no changes to numeric results, grouping, totals, logger configuration or default warning behavior.

## Review Focus

Renamed grouping columns must not produce invalid fit_by suggestions; prefitted learned binners must still be identified as learned; mixed global/stratified dimensions must report only global scope; failed fits must not report successful scope hints; notebook logger configuration must be restored.

## Task 1: Scoped informational logging and explain

Files: src/phl_risk/analysis/_binning_scope.py, _cube.py, _explain.py; tests/test_binning_scope_hints.py.
- [x] Test INFO severity, exact scope, filters, aliases, multiple groups, prefit learned bins, silence for fixed/fit_by/no grouping/compute/layout, failed fits, logging-state preservation, and explain global/stratified/fixed and totals semantics.
- [x] Observe failing tests, implement minimal helper and fit/explain integration, run focused tests.

## Task 2: Examples and verification

Files: examples/analysis_complete_guide.ipynb, docs/analysis_binning.md.
- [x] Add section 9.6 with selective logging, suppression checks, and explain; link it from guide navigation.
- [x] Execute full notebook and inspect outputs; run full pytest, ruff, diff checks and independent review.
- [x] Compare legacy numerical/metadata/layout snapshots again; only explain and opt-in logs may change.

## Execution

The existing managed checkout on work contains the prior feature; preserve it. User authorized implementation, so no further approval handoff is needed.

Verification: 542 passed, 8 skipped; notebook 42 code cells executed without errors. Legacy comparison: 48 scenarios, 632 DataFrames exactly equal, metadata/axes equal apart from timing. Independent review: no remaining correctness findings. A template-edge explain regression was reproduced and fixed; stratified plans report fitted strata rather than shared template edges. Ruff and diff checks passed.
