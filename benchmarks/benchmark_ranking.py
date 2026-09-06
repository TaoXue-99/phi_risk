"""Time the fixed AUC and KS adapters; no performance thresholds or SLA."""

import argparse
import json
from statistics import median
from timeit import repeat

import numpy as np
from sklearn import metrics as sklearn_metrics

from phl_risk.metrics import auc_score, ks_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.rows < 2 or args.repeat < 1:
        parser.error("rows >= 2 and repeat >= 1 are required")
    rng = np.random.default_rng(7)
    target = rng.integers(0, 2, args.rows)
    target[:2] = [0, 1]
    score = rng.random(args.rows)
    weight = rng.random(args.rows)

    expected_auc = sklearn_metrics.roc_auc_score(target, score, sample_weight=weight)
    fpr, tpr, _ = sklearn_metrics.roc_curve(
        target, score, sample_weight=weight, pos_label=1, drop_intermediate=False
    )
    np.testing.assert_allclose(auc_score(target, score, weight), expected_auc)
    np.testing.assert_allclose(ks_score(target, score, weight), np.max(np.abs(tpr - fpr)))
    timings = {
        name: median(repeat(lambda: kernel(target, score, weight), number=1, repeat=args.repeat))
        for name, kernel in (("auc_seconds", auc_score), ("ks_seconds", ks_score))
    }
    print(json.dumps({"rows": args.rows, "repeats": args.repeat, **timings}, indent=2))


if __name__ == "__main__":
    main()
