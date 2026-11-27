"""Bootstrap confidence intervals over queries.

Queries are the unit of sampling: a benchmark is a sample of the questions users
might ask, so resampling queries says how much a mean would move on another sample.
Both functions use a seeded generator, so a result file is reproducible.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    level: float = 0.95

    def as_dict(self, digits: int = 4) -> dict:
        return {
            "mean": round(self.mean, digits),
            "ci": [round(self.low, digits), round(self.high, digits)],
            "level": self.level,
        }


def _percentile_interval(samples: list[float], level: float) -> tuple[float, float]:
    ordered = sorted(samples)
    n = len(ordered)
    lo_idx = int(((1 - level) / 2) * n)
    hi_idx = min(n - 1, int(((1 + level) / 2) * n))
    return ordered[lo_idx], ordered[hi_idx]


def bootstrap_ci(
    values: list[float], rounds: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    low, high = _percentile_interval(means, level)
    return Interval(statistics.fmean(values), low, high, level)


def paired_bootstrap(
    a: list[float], b: list[float], rounds: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Interval for mean(b - a) over paired per-query scores (b better if positive)."""
    if len(a) != len(b):
        raise ValueError(f"paired scores must have equal length ({len(a)} != {len(b)})")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, rounds=rounds, level=level, seed=seed)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]
