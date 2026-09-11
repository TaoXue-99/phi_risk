"""Compare repeated pandas grouping with shared encoding; synthetic fixed-seed data.

Run: python benchmarks/benchmark_comparison.py --rows 100000 --fields 1 10 100
Add --rows 1000000 for the larger case. Reports medians; no speed SLA.
"""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd

from phl_risk.analysis import PSI, Cube, QuantileBinner
from phl_risk.metrics import psi_from_proportions


def naive(reference, current, dimensions, fields):
    """Fit each field once, but repeat pandas groupby/value_counts per field.

    This avoids an artificially slow per-group re-fit and keeps bin definitions
    mathematically equivalent to the optimized engine.
    """
    output = []
    for field in fields:
        binner = QuantileBinner(10).fit(reference[field])
        counts = []
        for frame in (reference, current):
            codes = binner.transform(frame[field]).cat.codes.to_numpy()
            if dimensions:
                scratch = frame[dimensions].copy()
                scratch["bin"] = codes
                table = (
                    scratch.groupby(dimensions, sort=False, observed=True)["bin"]
                    .value_counts(sort=False)
                    .unstack(fill_value=0)
                )
                table = table.reindex(columns=range(binner.n_bins_), fill_value=0)
            else:
                table = pd.DataFrame(
                    [
                        pd.Series(codes)
                        .value_counts()
                        .reindex(range(binner.n_bins_), fill_value=0)
                        .to_numpy()
                    ]
                )
            counts.append(table)
        a, b = counts
        keys = a.index.union(b.index, sort=False)
        a, b = a.reindex(keys, fill_value=0), b.reindex(keys, fill_value=0)
        p, q = a.to_numpy(dtype=float, copy=True), b.to_numpy(dtype=float, copy=True)
        p /= p.sum(axis=-1, keepdims=True)
        q /= q.sum(axis=-1, keepdims=True)
        output.append(psi_from_proportions(p, q))
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000])
    parser.add_argument("--fields", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    for rows in args.rows:
        for number in args.fields:
            rng = np.random.default_rng(42)
            fields = [f"x{i}" for i in range(number)]
            ref = pd.DataFrame(rng.normal(size=(rows, number)), columns=fields)
            cur = pd.DataFrame(rng.normal(0.3, size=(rows, number)), columns=fields)
            for i in range(max(args.dimensions, default=0)):
                ref[f"d{i}"] = rng.integers(0, 3, rows)
                cur[f"d{i}"] = rng.integers(0, 3, rows)
            for ndim in args.dimensions:
                dimensions = [f"d{i}" for i in range(ndim)]
                cube = Cube(dimensions, [PSI(field) for field in fields])
                timings = {"optimized": [], "pandas": []}
                for _ in range(args.repeats):
                    start = perf_counter()
                    result = cube.compute_comparison(ref, cur)
                    timings["optimized"].append(perf_counter() - start)
                    start = perf_counter()
                    expected = naive(ref, cur, dimensions, fields)
                    timings["pandas"].append(perf_counter() - start)
                    # pandas value_counts may sort group keys differently.
                    np.testing.assert_allclose(
                        np.sort(result.data_.value), np.sort(expected), rtol=1e-10, atol=1e-12
                    )
                print(
                    json.dumps(
                        {
                            "rows_per_side": rows,
                            "fields": number,
                            "dimensions": ndim,
                            "groups": result.metadata_["comparison_groups"],
                            "group_encoding_passes": result.metadata_["group_encoding_passes"],
                            "optimized_seconds": float(np.median(timings["optimized"])),
                            "pandas_seconds": float(np.median(timings["pandas"])),
                            "input_mib": float(
                                (
                                    ref.memory_usage(deep=True).sum()
                                    + cur.memory_usage(deep=True).sum()
                                )
                                / 2**20
                            ),
                            "repeats": args.repeats,
                        }
                    ),
                    flush=True,
                )
            del ref, cur


if __name__ == "__main__":
    main()
