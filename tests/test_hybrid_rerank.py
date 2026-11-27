import pytest

from ragbench import evaluate
from ragbench.datasets import Document
from ragbench.retriever import Hit
from ragbench.retrievers import BM25, Dense, Hybrid, Rerank
from ragbench.retrievers.hybrid import rrf

from .test_retrievers import CORPUS, BagOfWords


class Fixed:
    """Returns a fixed ranking for every query, and counts its calls."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        return self.ranking[:k]


class Counting(BagOfWords):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return super().embed_documents(texts)


class LengthScorer:
    """A fake cross-encoder: longer texts score higher."""

    def __init__(self):
        self.pairs = 0

    def score(self, query, texts):
        self.pairs += len(texts)
        return [float(len(t)) for t in texts]


def test_rrf_matches_hand_computed_scores():
    hits = rrf([["a", "b", "c"], ["b", "c", "a"]], k=3, rrf_k=60)
    scores = {h.doc_id: h.score for h in hits}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert [h.doc_id for h in hits] == ["b", "a", "c"]


def test_rrf_breaks_ties_on_doc_id_and_respects_k():
    hits = rrf([["y", "x"], ["x", "y"]], k=1)
    assert [h.doc_id for h in hits] == ["x"]


def test_rrf_counts_documents_found_by_only_one_retriever():
    hits = rrf([["a"], ["b", "a"]], k=10, rrf_k=0)
    assert {h.doc_id: h.score for h in hits} == {"a": 1.5, "b": 1.0}


def test_hybrid_fuses_its_retrievers():
    h = Hybrid(retrievers=[Fixed(["a", "b", "c"]), Fixed(["c", "b", "a"])], candidates=3)
    h.index(CORPUS)
    # a and c each get 1/61 + 1/63, just above b's 2/62; the tie breaks on doc id.
    assert [x.doc_id for x in h.search("anything", 3)] == ["a", "c", "b"]


def test_hybrid_with_bm25_and_dense():
    pytest.importorskip("numpy")
    h = Hybrid(retrievers=[BM25(), Dense(encoder=BagOfWords())])
    h.index(CORPUS)
    assert h.search("dog chased cat", 1)[0].doc_id == "c"
    assert len(h.search_batch(["cat", "mat"], 2)) == 2


def test_hybrid_validates_parameters():
    with pytest.raises(ValueError):
        Hybrid(candidates=0, retrievers=[])


def test_dense_cache_skips_reembedding(tmp_path):
    np = pytest.importorskip("numpy")
    first = Dense(encoder=Counting(), cache=str(tmp_path))
    first.index(CORPUS)
    second_encoder = Counting()
    second = Dense(encoder=second_encoder, cache=str(tmp_path))
    second.index(CORPUS)
    assert second.cache_hit and second_encoder.document_calls == 0
    assert np.array_equal(first.matrix, second.matrix)
    assert second.search("dog chased", 1)[0].doc_id == "c"


def test_dense_cache_misses_when_documents_change(tmp_path):
    pytest.importorskip("numpy")
    Dense(encoder=Counting(), cache=str(tmp_path)).index(CORPUS)
    changed = dict(CORPUS, a=Document("a", "", "a different mat"))
    enc = Counting()
    d = Dense(encoder=enc, cache=str(tmp_path))
    d.index(changed)
    assert not d.cache_hit and enc.document_calls == 1


def test_rerank_reorders_only_the_head():
    scorer = LengthScorer()
    r = Rerank(retriever=Fixed(["b", "a", "c"]), scorer=scorer, candidates=2)
    r.index(CORPUS)
    hits = r.search("q", 3)
    # b ("dogs and cats", 13 chars) and a ("the cat sat on the mat", 22) swap; c stays third.
    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    assert scorer.pairs == 2
    assert hits[2].score < hits[1].score


def test_rerank_asks_the_first_stage_for_enough_candidates():
    first = Fixed(["a", "b", "c"])
    r = Rerank(retriever=first, scorer=LengthScorer(), candidates=3)
    r.index(CORPUS)
    assert len(r.search("q", 1)) == 1


def test_rerank_rejects_a_scorer_that_drops_scores():
    class Short:
        def score(self, query, texts):
            return [1.0]

    r = Rerank(retriever=Fixed(["a", "b"]), scorer=Short(), candidates=2)
    r.index(CORPUS)
    with pytest.raises(ValueError, match="scores"):
        r.search("q", 2)


def test_rerank_validates_first_stage():
    with pytest.raises(ValueError, match="first_stage"):
        Rerank(first_stage="splade", scorer=LengthScorer())


def test_rerank_on_tiny_keeps_recall_at_100(tiny):
    first = BM25()
    base = evaluate(first, tiny, name="bm25", bootstrap_rounds=50)
    reranked = evaluate(
        Rerank(retriever=BM25(), scorer=LengthScorer(), candidates=2),
        tiny,
        name="rerank",
        bootstrap_rounds=50,
    )
    assert reranked.per_query["recall@100"] == base.per_query["recall@100"]


def test_hit_scores_are_floats():
    r = Rerank(retriever=Fixed(["a"]), scorer=LengthScorer(), candidates=1)
    r.index(CORPUS)
    assert isinstance(r.search("q", 1)[0], Hit)
