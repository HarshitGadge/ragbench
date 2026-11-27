"""Dense retrieval: embed the corpus once, rank by cosine similarity.

The default encoder is `fastembed <https://github.com/qdrant/fastembed>`_ (ONNX, CPU),
installed with ``pip install 'ragbench-eval[dense]'``. Pass ``encoder=`` to use anything else:
an object with ``embed_documents(texts)`` and ``embed_query(text)`` returning vectors.

Embedding a corpus is the slow part of a dense run. Pass ``cache="some/dir"`` to save the
corpus matrix there and reuse it on the next run with the same model and documents, for
example when the same embeddings feed ``Dense``, ``Hybrid`` and ``Rerank``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragbench.datasets import Document
from ragbench.retriever import Hit


class FastEmbedEncoder:
    def __init__(self, model: str, batch_size: int = 256, query_prefix: str = ""):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("Dense needs: pip install 'ragbench-eval[dense]'") from exc
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
        cache: str | None = None,
    ):
        self.model_name = model
        self.encoder = encoder or FastEmbedEncoder(model, batch_size, query_prefix)
        self.cache = Path(cache) if cache else None
        self.doc_ids: list[str] = []
        self.matrix = None
        self.cache_hit = False

    def _cache_path(self, corpus: Mapping[str, Document]) -> Path:
        h = hashlib.sha256(self.model_name.encode())
        for doc_id in self.doc_ids:
            h.update(b"\0" + doc_id.encode() + b"\t" + corpus[doc_id].text.encode())
        return self.cache / f"{h.hexdigest()[:32]}.npy"

    @staticmethod
    def _normalise(matrix):
        import numpy as np

        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def index(self, corpus: Mapping[str, Document]) -> None:
        import numpy as np

        self.doc_ids = list(corpus)
        path = self._cache_path(corpus) if self.cache else None
        if path is not None and path.exists():
            self.matrix = np.load(path)
            self.cache_hit = True
            return
        vectors = self.encoder.embed_documents([corpus[d].text for d in self.doc_ids])
        self.matrix = self._normalise(vectors)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, self.matrix)
            tmp.replace(path)

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
