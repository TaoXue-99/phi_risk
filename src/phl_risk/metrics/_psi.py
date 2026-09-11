"""PSI on aligned population counts; smoothing applies to every bucket."""

import numpy as np


def psi_score(reference_counts, current_counts, epsilon=1e-6):
    ref, cur = np.asarray(reference_counts, dtype=float), np.asarray(current_counts, dtype=float)
    if ref.ndim != 1 or ref.shape != cur.shape or not ref.size:
        raise ValueError("PSI requires nonempty, aligned one-dimensional counts")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if (
        not np.isfinite(ref).all()
        or not np.isfinite(cur).all()
        or (ref < 0).any()
        or (cur < 0).any()
    ):
        raise ValueError("PSI counts must be nonnegative and finite")
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("PSI requires nonempty populations")
    p, q = np.maximum(ref / ref.sum(), epsilon), np.maximum(cur / cur.sum(), epsilon)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))
