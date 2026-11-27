"""Okapi BM25 in the standard library.

Scoring follows Lucene's BM25 (the one Elasticsearch, Anserini and Pyserini use):

    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(d, q) = sum over query terms t of
                  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

Defaults k1=0.9, b=0.4 are Anserini's. Tokenisation is lowercase word characters,
with optional English stop-word removal and optional Snowball stemming
(``pip install 'ragbench[stem]'``). Scores will differ a little from Elasticsearch-based
BEIR baselines because analysers differ.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping

from ragbench.datasets import Document
from ragbench.retriever import Hit

TOKEN = re.compile(r"\w+", re.UNICODE)

# Lucene's default English stop words.
STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the their
    then there these they this to was will with""".split()  # noqa: SIM905
)


def make_tokenizer(stopwords: bool = True, stem: bool = False) -> Callable[[str], list[str]]:
    stemmer = None
    if stem:
        try:
            import snowballstemmer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("stem=True needs: pip install 'ragbench[stem]'") from exc
        stemmer = snowballstemmer.stemmer("english")

    def tokenize(text: str) -> list[str]:
        tokens = TOKEN.findall(text.lower())
        if stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if stemmer is not None:
            tokens = stemmer.stemWords(tokens)
        return tokens

    return tokenize


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4, stopwords: bool = True, stem: bool = False):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.tokenize = make_tokenizer(stopwords=stopwords, stem=stem)
        self.doc_ids: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.norm: list[float] = []

    def index(self, corpus: Mapping[str, Document]) -> None:
        self.doc_ids = list(corpus)
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for i, doc_id in enumerate(self.doc_ids):
            tokens = self.tokenize(corpus[doc_id].text)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((i, tf))
        n = len(self.doc_ids)
        avgdl = (sum(lengths) / n) if n else 0.0
        self.postings = postings
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
        }
        # Per-document length normalisation k1 * (1 - b + b * |d| / avgdl), precomputed.
        self.norm = [
            self.k1 * (1 - self.b + self.b * (length / avgdl if avgdl else 0.0))
            for length in lengths
        ]

    def scores(self, query: str) -> dict[int, float]:
        acc: dict[int, float] = {}
        for term, qtf in Counter(self.tokenize(query)).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, tf in postings:
                acc[i] = acc.get(i, 0.0) + qtf * idf * tf * (self.k1 + 1) / (tf + self.norm[i])
        return acc

    def search(self, query: str, k: int) -> list[Hit]:
        acc = self.scores(query)
        # Ties broken by document id so runs are deterministic.
        top = heapq.nsmallest(k, acc.items(), key=lambda x: (-x[1], self.doc_ids[x[0]]))
        return [Hit(self.doc_ids[i], s) for i, s in top]
