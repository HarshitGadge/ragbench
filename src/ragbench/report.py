"""Render result files as a Markdown table, so README numbers are generated, not typed."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.runner import RunResult

DEFAULT_METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10")


def markdown_table(runs: Sequence[RunResult], metrics: Sequence[str] = DEFAULT_METRICS) -> str:
    header = ["dataset", "retriever", *metrics, "queries", "p50 ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in runs:
        summary = run.summary(
            rounds=run.settings.get("bootstrap_rounds", 2000), seed=run.settings.get("seed", 0)
        )
        cells = [run.dataset.get("name", "?"), run.name]
        for metric in metrics:
            if metric in summary:
                s = summary[metric]
                cells.append(f"{s['mean']:.4f} ({s['ci'][0]:.3f} to {s['ci'][1]:.3f})")
            else:
                cells.append("-")
        cells += [str(run.dataset.get("evaluated_queries", "?")), f"{run.latency_ms['p50']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
