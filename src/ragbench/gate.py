"""Pass or fail a candidate run against a baseline, for use as a CI check.

A gate compares two runs on the same queries with the paired bootstrap from
:mod:`ragbench.compare` and fails when a metric dropped by more than an
allowed amount. Two modes decide what "dropped" means:

``mean``
    Fail when the mean change is below ``-max_drop``. Simple and strict, so a
    noisy run can fail by chance.
``ci``
    Fail only when the whole 95% interval for the change is below
    ``-max_drop``, so the drop is unlikely to be noise. Small query sets will
    rarely fail in this mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragbench.compare import Comparison, compare_runs
from ragbench.runner import RunResult

MODES = ("mean", "ci")
# Settings that change what a score means. Runs that differ on any of these are
# not comparable, so the gate refuses them rather than report a false result.
MUST_MATCH = ("depth", "ndcg_k", "ignore_identical_ids")


@dataclass(frozen=True)
class Check:
    comparison: Comparison
    max_drop: float
    mode: str

    @property
    def passed(self) -> bool:
        d = self.comparison.diff
        edge = d.high if self.mode == "ci" else d.mean
        return edge >= -self.max_drop

    def line(self) -> str:
        c, d = self.comparison, self.comparison.diff
        verdict = "pass" if self.passed else "FAIL"
        return (
            f"{verdict} {c.metric}: {c.baseline_mean:.4f} -> {c.candidate_mean:.4f}, "
            f"change {d.mean:+.4f} (95% CI {d.low:+.4f} to {d.high:+.4f}, "
            f"{c.queries} queries), allowed drop {self.max_drop:.4f}, mode {self.mode}"
        )


@dataclass(frozen=True)
class GateResult:
    checks: list[Check]
    dataset: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def markdown(self) -> str:
        head = "passed" if self.passed else "failed"
        rows = [
            f"### ragbench gate {head} on {self.dataset}",
            "",
            "| metric | baseline | candidate | change (95% CI) | allowed drop | result |",
            "|---|---|---|---|---|---|",
        ]
        for ch in self.checks:
            c, d = ch.comparison, ch.comparison.diff
            rows.append(
                f"| {c.metric} | {c.baseline_mean:.4f} | {c.candidate_mean:.4f} | "
                f"{d.mean:+.4f} ({d.low:+.4f} to {d.high:+.4f}) | {ch.max_drop:.4f} "
                f"({ch.mode}) | {'pass' if ch.passed else 'fail'} |"
            )
        queries = self.checks[0].comparison.queries if self.checks else 0
        rows += ["", f"{queries} queries, paired bootstrap."]
        return "\n".join(rows) + "\n"


def _check_settings(baseline: RunResult, candidate: RunResult) -> None:
    a, b = baseline.dataset.get("split"), candidate.dataset.get("split")
    if a != b:
        raise ValueError(f"runs are on different splits ({a!r} vs {b!r})")
    for key in MUST_MATCH:
        a, b = baseline.settings.get(key), candidate.settings.get(key)
        if a != b:
            raise ValueError(f"runs differ on {key} ({a!r} vs {b!r}), so they aren't comparable")


def gate(
    baseline: RunResult,
    candidate: RunResult,
    metrics: list[str] | tuple[str, ...] = ("ndcg@10",),
    max_drop: float = 0.01,
    mode: str = "mean",
    rounds: int = 2000,
    seed: int = 0,
) -> GateResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if max_drop < 0:
        raise ValueError("max_drop can't be negative")
    _check_settings(baseline, candidate)
    checks = [
        Check(compare_runs(baseline, candidate, m, rounds=rounds, seed=seed), max_drop, mode)
        for m in metrics
    ]
    return GateResult(checks=checks, dataset=str(baseline.dataset.get("name")))
