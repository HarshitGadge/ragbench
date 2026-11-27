"""Hybrid retrieval: fuse several rankings with reciprocal rank fusion (RRF).

Each retriever returns its top ``candidates`` documents, and a document's fused
score is the sum over retrievers of ``1 / (rrf_k + rank)``, with ranks from 1.
RRF uses ranks only, so BM25 and cosine scores need no normalising. ``rrf_k=60``
is the constant from Cormack, Clarke and Buettcher (SIGIR 2009).

By default it fuses BM25 (stemmed) with the dense retriever::

    ragbench run --data data/scifact --retriever ragbench.retrievers:Hybrid \\
        --param cache=cache/embeddings
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragbench.datasets import Document
from ragbench.retriever import Hit, doc_ids, index_if_supported
from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import Dense


def rrf(rankings: Sequence[Sequence[str]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Fuse ranked lists of doc ids; ties break on doc id so runs are deterministic."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda s: (-s[1], s[0]))
    return [Hit(doc_id, score) for doc_id, score in ordered[:k]]


def _batch(retriever: Any, queries: Sequence[str], k: int) -> list[list[str]]:
    batch = getattr(retriever, "search_batch", None)
    results = (
        batch(list(queries), k) if callable(batch) else [retriever.search(q, k) for q in queries]
    )
    return [doc_ids(r) for r in results]


class Hybrid:
    def __init__(
        self,
        rrf_k: int = 60,
        candidates: int = 100,
        stem: bool = True,
        model: str = "BAAI/bge-small-en-v1.5",
        cache: str | None = None,
        retrievers: Sequence[Any] | None = None,
    ):
        if rrf_k < 0 or candidates < 1:
            raise ValueError("rrf_k must be >= 0 and candidates >= 1")
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.retrievers = (
            list(retrievers)
            if retrievers is not None
            else [BM25(stem=stem), Dense(model=model, cache=cache)]
        )

    def index(self, corpus: Mapping[str, Document]) -> None:
        for r in self.retrievers:
            index_if_supported(r, corpus)

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        depth = max(k, self.candidates)
        per_retriever = [_batch(r, queries, depth) for r in self.retrievers]
        return [
            rrf([lists[i] for lists in per_retriever], k, self.rrf_k) for i in range(len(queries))
        ]
