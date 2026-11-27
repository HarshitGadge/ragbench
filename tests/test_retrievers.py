import math

import pytest

from ragbench import evaluate
from ragbench.datasets import Document
from ragbench.retriever import load_retriever
from ragbench.retrievers import BM25, Dense
from ragbench.retrievers.bm25 import make_tokenizer

CORPUS = {
    "a": Document("a", "", "the cat sat on the mat"),
    "b": Document("b", "", "dogs and cats"),
    "c": Document("c", "", "a dog chased a cat a cat"),
}


def test_tokenizer():
    assert make_tokenizer()("The Cat, the HAT!") == ["cat", "hat"]
    assert make_tokenizer(stopwords=False)("The cat") == ["the", "cat"]


def test_bm25_matches_hand_computed_lucene_score():
    k1, b = 1.2, 0.75
    bm = BM25(k1=k1, b=b)
    bm.index(CORPUS)
    # after stop words: a=[cat, sat, mat] (3), b=[dogs, cats] (2), c=[dog, chased, cat, cat] (4)
    avgdl = (3 + 2 + 4) / 3
    n, df = 3, 2  # "cat" is in a and c
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def term(tf, dl):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    hits = {h.doc_id: h.score for h in bm.search("cat", 10)}
    assert hits["a"] == pytest.approx(term(1, 3))
    assert hits["c"] == pytest.approx(term(2, 4))
    assert "b" not in hits  # "cats" is a different token without stemming


def test_bm25_ranking_and_ties_are_deterministic():
    bm = BM25()
    bm.index(CORPUS)
    assert [h.doc_id for h in bm.search("dog chased cat", 3)][0] == "c"
    assert bm.search("unseen words only", 5) == []
    bm2 = BM25()
    bm2.index({"y": Document("y", "", "same"), "x": Document("x", "", "same")})
    assert [h.doc_id for h in bm2.search("same", 2)] == ["x", "y"]


def test_bm25_validates_parameters():
    with pytest.raises(ValueError):
        BM25(b=1.5)


def test_bm25_on_tiny(tiny):
    result = evaluate(BM25(), tiny, bootstrap_rounds=100)
    assert result.per_query["recall@10"]["q2"] == 1.0
    assert 0 < result.mean("ndcg@10") <= 1


def test_bm25_loads_from_cli_spec():
    r = load_retriever("ragbench.retrievers:BM25", {"k1": 1.5})
    assert r.k1 == 1.5


class BagOfWords:
    """A fake encoder: one dimension per vocabulary word. No model download."""

    vocab = ["cat", "dog", "mat", "chased", "sat"]

    def _vec(self, text):
        words = text.lower().split()
        return [float(sum(w.startswith(v) for w in words)) for v in self.vocab]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_dense_with_custom_encoder():
    np = pytest.importorskip("numpy")
    d = Dense(encoder=BagOfWords())
    d.index(CORPUS)
    hits = d.search("dog chased", 2)
    assert hits[0].doc_id == "c"
    assert len(d.search_batch(["cat", "mat sat"], 10)[1]) == 3
    assert d.search("mat sat", 1)[0].doc_id == "a"
    assert np.allclose(np.linalg.norm(d.matrix, axis=1), 1.0)


def test_dense_requires_index():
    pytest.importorskip("numpy")
    with pytest.raises(RuntimeError):
        Dense(encoder=BagOfWords()).search("cat", 1)


def test_bm25_stemming_matches_plural_forms():
    pytest.importorskip("snowballstemmer")
    bm = BM25(stem=True)
    bm.index(CORPUS)
    assert {h.doc_id for h in bm.search("cat", 10)} == {"a", "b", "c"}
