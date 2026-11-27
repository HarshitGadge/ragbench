"""Two-stage retrieval: a first-stage retriever, then a cross-encoder reranker.

The first stage returns its top ``depth`` documents. The cross-encoder scores
each (query, document) pair for the top ``candidates`` of them, and those are
reordered by that score. Anything past ``candidates`` keeps its first-stage
order, so Recall@100 can only change when ``candidates`` reaches 100.

The default scorer is fastembed's ``TextCrossEncoder`` (ONNX, CPU), installed
with ``pip install 'ragbench[dense]'``. Pass ``scorer=`` to use anything else: an
object with ``score(query, texts)`` returning one float per text.

    ragbench run --data data/scifact --retriever ragbench.retrievers:Rerank \\
        --param first_stage=hybrid --param candidates=30
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragbench.datasets import Document
from ragbench.retriever import Hit, index_if_supported
from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import Dense
from ragbench.retrievers.hybrid import Hybrid, _batch

FIRST_STAGES = ("bm25", "dense", "hybrid")


class FastEmbedCrossEncoder:
    def __init__(self, model: str, batch_size: int = 32):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("Rerank needs: pip install 'ragbench[dense]'") from exc
        self.model = TextCrossEncoder(model_name=model)
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, list(texts), batch_size=self.batch_size)]


def _first_stage(name: str, stem: bool, model: str, cache: str | None) -> Any:
    if name == "bm25":
        return BM25(stem=stem)
    if name == "dense":
        return Dense(model=model, cache=cache)
    if name == "hybrid":
        return Hybrid(stem=stem, model=model, cache=cache)
    raise ValueError(f"first_stage must be one of {FIRST_STAGES}, not {name!r}")


class Rerank:
    def __init__(
        self,
        first_stage: str = "hybrid",
        model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        candidates: int = 30,
        stem: bool = True,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        cache: str | None = None,
        batch_size: int = 32,
        scorer: Any = None,
        retriever: Any = None,
    ):
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self.model_name = model
        self.candidates = candidates
        self.first = (
            retriever
            if retriever is not None
            else _first_stage(first_stage, stem, dense_model, cache)
        )
        self.scorer = scorer or FastEmbedCrossEncoder(model, batch_size)
        self.texts: dict[str, str] = {}

    def index(self, corpus: Mapping[str, Document]) -> None:
        self.texts = {doc_id: doc.text for doc_id, doc in corpus.items()}
        index_if_supported(self.first, corpus)

    def _rerank(self, query: str, ranking: list[str], k: int) -> list[Hit]:
        head, tail = ranking[: self.candidates], ranking[self.candidates :]
        scores = self.scorer.score(query, [self.texts[d] for d in head]) if head else []
        if len(scores) != len(head):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(head)} texts")
        ordered = sorted(zip(head, scores, strict=True), key=lambda s: (-s[1], s[0]))
        hits = [Hit(d, float(s)) for d, s in ordered]
        # Documents past the reranked head keep their first-stage order, scored below it.
        floor = min((h.score for h in hits), default=0.0)
        hits += [Hit(d, floor - i - 1.0) for i, d in enumerate(tail)]
        return hits[:k]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        depth = max(k, self.candidates)
        rankings = _batch(self.first, queries, depth)
        return [self._rerank(q, r, k) for q, r in zip(queries, rankings, strict=True)]
