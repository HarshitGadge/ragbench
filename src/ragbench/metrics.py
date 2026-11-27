"""Per-query ranking metrics.

Each function takes a ranking (document ids, best first, no duplicates) and the
query's judgements. nDCG uses linear gains (the grade itself) and log2 discounts,
which is what ``trec_eval``/``pytrec_eval`` and therefore BEIR report.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def dedupe(doc_ids: Iterable[str]) -> list[str]:
    """Keep the first occurrence of each id, so a document counts once, at its best rank.

    This is how chunk-level hits become a document ranking.
    """
    seen: set[str] = set()
    ranking: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def dcg_at_k(ranking: list[str], grades: Mapping[str, int], k: int) -> float:
    return sum(grades.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))


def ndcg_at_k(ranking: list[str], grades: Mapping[str, int] | set[str], k: int = 10) -> float:
    """Normalised DCG at ``k``. A set of ids is treated as binary judgements."""
    if isinstance(grades, set):
        grades = dict.fromkeys(grades, 1)
    ideal_order = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_order))
    return dcg_at_k(ranking, grades, k) / ideal if ideal else 0.0


def recall_at_k(ranking: list[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def mrr_at_k(ranking: list[str], relevant: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant)
    return next((1 / (i + 1) for i, d in enumerate(ranking[:k]) if d in relevant), 0.0)


def precision_at_k(ranking: list[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return len([d for d in ranking[:k] if d in relevant]) / k


def score_query(
    ranking: list[str], grades: Mapping[str, int], ndcg_k: int = 10, recall_ks=(10, 100)
) -> dict[str, float]:
    """All the per-query metrics RAGBench reports, keyed by name (e.g. ``ndcg@10``)."""
    relevant = set(grades)
    scores = {
        f"ndcg@{ndcg_k}": ndcg_at_k(ranking, grades, ndcg_k),
        f"mrr@{ndcg_k}": mrr_at_k(ranking, relevant, ndcg_k),
    }
    for k in recall_ks:
        scores[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
    return scores
