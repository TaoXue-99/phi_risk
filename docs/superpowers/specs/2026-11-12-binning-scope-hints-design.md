# Binning scope hints

The user accepted INFO logging plus explain() after discussing warnings versus explanatory logs, and authorized implementation and notebook examples.

- Preserve all fitting, computation and totals behavior. Do not configure library logging handlers or levels.
- After successful Cube.fit, emit one English INFO record per global BinDimension when ordinary grouping dimensions exist. Emit no scope hint in compute/layout, for explicit fit_by, FixedBinner, or a cube without ordinary grouping.
- Use the prepared fit reference wording: Cube filters and missing-stratum policies can already restrict it. Suggest valid raw, unrenamed grouping fields via fit_by; when only renamed grouping axes exist, explain the raw-column prerequisite rather than suggesting invalid arguments.
- Add explicit global/within-group/fixed scope descriptions to explain(). Stratified totals descriptions must state the retained fit_by fields; legacy totals wording remains unchanged.
- Logger event analysis.binning.global_scope allows selective capture. INFO is hidden unless the application enables it; do not emit UserWarning for valid configurations.
- Notebook section 9.6 demonstrates temporary, selective logging with settings restored, suppression for fit_by/compute/layout/fixed boundaries, and explain() before fitting. The guide's narrative remains Chinese; runtime hints are English.
