"""Shared presentation of full-precision, strictly increasing bin edges."""

from collections.abc import Sequence


def format_intervals(
    edges: Sequence[float], *, precision: int = 6, include_lowest: bool = True
) -> tuple[str, ...]:
    """Format decimal places, increasing precision to distinguish nearby edges.

    Float64 boundaries need at most 324 decimal places, including subnormals.
    Formatting never changes the supplied boundaries or assignment semantics.
    """
    for digits in range(precision, max(precision, 324) + 1):
        formatted = tuple(f"{edge:.{digits}f}" for edge in edges)
        # Numeric comparison also catches -0.00 versus 0.00 collisions.
        if all(float(left) < float(right) for left, right in zip(formatted, formatted[1:])):
            break
    return tuple(
        f"{'[' if i == 0 and include_lowest else '('}{left}, {right}]"
        for i, (left, right) in enumerate(zip(formatted, formatted[1:]))
    )
