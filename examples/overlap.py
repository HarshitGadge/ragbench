"""A deliberately simple retriever: rank documents by how many query words they share.

It exists to show the plug-in interface, not to be good. Run it with:

    ragbench run --data tests/fixtures/tiny --retriever examples/overlap.py:WordOverlap
"""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


class WordOverlap:
    def __init__(self, min_overlap: int = 1):
        self.min_overlap = min_overlap
        self.docs: dict[str, set[str]] = {}

    def index(self, corpus):
        self.docs = {doc_id: words(doc.text) for doc_id, doc in corpus.items()}

    def search(self, query: str, k: int):
        q = words(query)
        scored = [(doc_id, len(q & terms)) for doc_id, terms in self.docs.items()]
        scored = [s for s in scored if s[1] >= self.min_overlap]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]
