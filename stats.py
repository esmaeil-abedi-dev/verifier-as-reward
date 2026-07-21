"""
stats.py
========

Small statistics helpers shared across the revision experiments. The only
non-trivial piece is the Wilson score interval for a binomial proportion,
which the reviewer asked for on every accuracy / false-authorize number.

Wilson is used rather than the normal (Wald) interval because our per-class
and small-set counts are exactly the regime where Wald misbehaves (it gives
[0,0] at k=0 and over-covers near the boundaries). Wilson stays within [0,1]
and is well-behaved for small n and extreme proportions.

No third-party dependency; pure math.
"""

from __future__ import annotations

import math
from typing import Optional


def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[tuple]:
    """95% (default z=1.96) Wilson score interval for k successes in n trials.

    Returns (lo, hi) clamped to [0, 1], or None if n == 0. The point estimate
    is k/n; the interval is centered on the Wilson-adjusted proportion, not
    on k/n, which is why lo/hi are not symmetric about k/n."""
    if n == 0:
        return None
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range for n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_with_ci(k: int, n: int, z: float = 1.96) -> dict:
    """A proportion reported with its count and Wilson interval:
    {rate, n, k, ci: [lo, hi]} — or rate/ci None when n == 0."""
    ci = wilson_ci(k, n, z)
    return {
        "rate": (k / n) if n else None,
        "n": n,
        "k": k,
        "ci": list(ci) if ci else None,
    }


def fmt_pct_ci(k: int, n: int, z: float = 1.96) -> str:
    """Human-readable 'xx.x% [lo, hi] (k/n)' for logs and tables."""
    if n == 0:
        return "n/a (0)"
    lo, hi = wilson_ci(k, n, z)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"
