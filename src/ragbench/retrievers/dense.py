"""Dense retrieval: embed the corpus once, rank by cosine similarity.

The default encoder is `fastembed <https://github.com/qdrant/fastembed>`_ (ONNX, CPU),
installed with ``pip install 'ragbench[dense]'``. Pass ``encoder=`` to use anything else:
an object with ``embed_documents(texts)`` and ``embed_query(text)`` returning vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragbench.datasets import Document
from ragbench.retriever import Hit


class FastEmbedEncoder:
    def __init__(self, model: str, batch_size: int = 256, query_prefix: str = ""):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("Dense needs: pip install 'ragbench[dense]'") from exc
        self.model = TextEmbedding(model_name=model)
        self.batch_size = batch_size
        self.query_prefix = query_prefix

    def embed_documents(self, texts: Sequence[str]):
        return list(self.model.embed(list(texts), batch_size=self.batch_size))

    def embed_queries(self, texts: Sequence[str]):
        return list(self.model.embed([self.query_prefix + t for t in texts], batch_size=64))

    def embed_query(self, text: str):
        return self.embed_queries([text])[0]


class Dense:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "",
        batch_size: int = 256,
        encoder: Any = None,
    ):
        self.model_name = model
        self.encoder = encoder or FastEmbedEncoder(model, batch_size, query_prefix)
        self.doc_ids: list[str] = []
        self.matrix = None

    @staticmethod
    def _normalise(matrix):
        import numpy as np

        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def index(self, corpus: Mapping[str, Document]) -> None:
        self.doc_ids = list(corpus)
        vectors = self.encoder.embed_documents([corpus[d].text for d in self.doc_ids])
        self.matrix = self._normalise(vectors)

    def _top(self, sims, k: int) -> list[Hit]:
        import numpy as np

        k = min(k, len(self.doc_ids))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = sorted(idx, key=lambda i: (-float(sims[i]), self.doc_ids[i]))
        return [Hit(self.doc_ids[i], float(sims[i])) for i in idx]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        if self.matrix is None:
            raise RuntimeError("Dense.index() must run before search")
        embed = getattr(self.encoder, "embed_queries", None)
        vectors = embed(list(queries)) if embed else [self.encoder.embed_query(q) for q in queries]
        sims = self._normalise(vectors) @ self.matrix.T
        return [self._top(row, k) for row in sims]
