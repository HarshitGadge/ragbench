"""Run a retriever over a dataset and keep per-query scores, so runs can be compared."""

from __future__ import annotations

import datetime as dt
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragbench.datasets import Dataset
from ragbench.metrics import dedupe, score_query
from ragbench.retriever import doc_ids, index_if_supported
from ragbench.stats import bootstrap_ci, percentile

SCHEMA_VERSION = 1


@dataclass
class RunResult:
    """Everything one run produced. ``per_query[metric][query_id]`` holds raw scores."""

    name: str
    dataset: dict
    settings: dict
    per_query: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    index_seconds: float | None
    meta: dict = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        return statistics.fmean(self.per_query[metric].values())

    def summary(self, rounds: int = 2000, seed: int = 0) -> dict[str, dict]:
        out = {}
        for metric, scores in self.per_query.items():
            out[metric] = bootstrap_ci(list(scores.values()), rounds=rounds, seed=seed).as_dict()
        return out

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "dataset": self.dataset,
            "settings": self.settings,
            "metrics": self.summary(
                rounds=self.settings.get("bootstrap_rounds", 2000),
                seed=self.settings.get("seed", 0),
            ),
            "latency_ms": self.latency_ms,
            "index_seconds": self.index_seconds,
            "meta": self.meta,
            "per_query": {m: dict(sorted(s.items())) for m, s in self.per_query.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported result schema {data.get('schema')!r}")
        return cls(
            name=data["name"],
            dataset=data["dataset"],
            settings=data["settings"],
            per_query=data["per_query"],
            latency_ms=data["latency_ms"],
            index_seconds=data["index_seconds"],
            meta=data.get("meta", {}),
        )


def evaluate(
    retriever: Any,
    dataset: Dataset,
    *,
    name: str = "retriever",
    depth: int = 100,
    ndcg_k: int = 10,
    recall_ks: tuple[int, ...] = (10, 100),
    limit: int | None = None,
    bootstrap_rounds: int = 2000,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    meta: dict | None = None,
) -> RunResult:
    """Index the corpus (if the retriever supports it), run every query, score it."""
    if depth < max((ndcg_k, *recall_ks)):
        raise ValueError(f"depth {depth} is shallower than the deepest cutoff")

    started = time.perf_counter()
    index_if_supported(retriever, dataset.corpus)
    index_seconds = time.perf_counter() - started

    query_items = list(dataset.queries.items())[:limit] if limit else list(dataset.queries.items())
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []

    batch = getattr(retriever, "search_batch", None)
    if callable(batch):
        t0 = time.perf_counter()
        results = batch([text for _, text in query_items], depth)
        per_query_ms = (time.perf_counter() - t0) * 1000 / max(1, len(query_items))
        for (qid, _), result in zip(query_items, results, strict=True):
            rankings[qid] = dedupe(doc_ids(result))[:depth]
            latencies.append(per_query_ms)
    else:
        for i, (qid, text) in enumerate(query_items, start=1):
            t0 = time.perf_counter()
            result = retriever.search(text, depth)
            latencies.append((time.perf_counter() - t0) * 1000)
            rankings[qid] = dedupe(doc_ids(result))[:depth]
            if progress:
                progress(i, len(query_items))

    per_query: dict[str, dict[str, float]] = {}
    for qid, ranking in rankings.items():
        for metric, value in score_query(ranking, dataset.qrels[qid], ndcg_k, recall_ks).items():
            per_query.setdefault(metric, {})[qid] = value

    latency = {
        "p50": round(percentile(latencies, 50), 3),
        "p95": round(percentile(latencies, 95), 3),
        "mean": round(statistics.fmean(latencies), 3),
        "batched": callable(batch),
    }
    from ragbench import __version__

    run_meta = {
        "ragbench": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    } | (meta or {})
    return RunResult(
        name=name,
        dataset=dataset.summary() | {"evaluated_queries": len(rankings)},
        settings={
            "depth": depth,
            "ndcg_k": ndcg_k,
            "recall_ks": list(recall_ks),
            "limit": limit,
            "bootstrap_rounds": bootstrap_rounds,
            "seed": seed,
        },
        per_query=per_query,
        latency_ms=latency,
        index_seconds=round(index_seconds, 3),
        meta=run_meta,
    )
