# Method initialization implementation plan

Approved user design. Execute inline using executing-plans; preserve unrelated working changes.

- [x] Add initialization module, backend-owned template, thin MethodExperiment and explicit open.
- [x] File-locked persistent method numbering, UTC directory time, project-wide record lookup.
- [x] Explicit YAML input and independent Hydra composition; capture source bytes and resolved facts.
- [x] Compact comparison (run/time/metrics/dict params), separate recursive config comparison.
- [x] Tests: idempotent/no-write open, numbering across processes/failures, input provenance,
  immutable snapshots, generic record independence, comparison missing-vs-null semantics.
- [x] Rewrite script and 100k/60-feature guide, execute and save outputs, update docs/version notes.
- [x] Full pytest, lower LightGBM compatibility, Ruff and build checks.

Record schema stays 3: new optional run label/method metadata is additive. Existing root Runs
remain readable. New initialized methods use method.json, configs, reports, runs and sequence.json.
No code duplication in training or persistence. PyYAML is a LightGBM extra; filelock protects
cross-process sequence allocation. File configuration is read once, resolved once, never re-read
when serializing the run. Native dict/config APIs continue to work. No auto-selection.
