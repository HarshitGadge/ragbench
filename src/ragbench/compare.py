"""Compare two runs on the same queries with a paired bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

from ragbench.runner import RunResult
from ragbench.stats import Interval, paired_bootstrap


@dataclass(frozen=True)
class Comparison:
    metric: str
    baseline: str
    candidate: str
    baseline_mean: float
    candidate_mean: float
    diff: Interval
    queries: int

    @property
    def significant(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.diff.low > 0 or self.diff.high < 0

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "baseline_mean": round(self.baseline_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "diff": self.diff.as_dict(),
            "queries": self.queries,
            "significant": self.significant,
        }


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    metric: str = "ndcg@10",
    rounds: int = 2000,
    seed: int = 0,
) -> Comparison:
    if baseline.dataset.get("name") != candidate.dataset.get("name"):
        raise ValueError(
            f"runs are on different datasets: {baseline.dataset.get('name')} "
            f"vs {candidate.dataset.get('name')}"
        )
    for run in (baseline, candidate):
        if metric not in run.per_query:
            raise KeyError(f"{run.name} has no {metric!r}; it has {sorted(run.per_query)}")
    a, b = baseline.per_query[metric], candidate.per_query[metric]
    if set(a) != set(b):
        raise ValueError(
            f"runs scored different queries ({len(a)} vs {len(b)}, {len(set(a) ^ set(b))} differ)"
        )
    qids = sorted(a)
    xs, ys = [a[q] for q in qids], [b[q] for q in qids]
    return Comparison(
        metric=metric,
        baseline=baseline.name,
        candidate=candidate.name,
        baseline_mean=sum(xs) / len(xs),
        candidate_mean=sum(ys) / len(ys),
        diff=paired_bootstrap(xs, ys, rounds=rounds, seed=seed),
        queries=len(qids),
    )
