import math
import random

import pytest

from ragbench.metrics import (
    dedupe,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_query,
)


def test_dedupe_keeps_best_rank():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_ndcg_perfect_and_empty():
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], grades) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], grades) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_ndcg_hand_computed_graded():
    # ranking b(1), x(0), a(2); ideal a(2), b(1)
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "x", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_cutoff_ignores_later_ranks():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_ndcg_binary_matches_original_eval_code():
    # The formula from agentic-rag-service/eval/retrieval_eval.py (binary gains).
    def original(ranking, relevant, k=10):
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / ideal if ideal else 0.0

    rng = random.Random(7)
    docs = [f"d{i}" for i in range(40)]
    for _ in range(200):
        relevant = set(rng.sample(docs, rng.randint(1, 12)))
        ranking = rng.sample(docs, 25)
        assert ndcg_at_k(ranking, relevant) == pytest.approx(original(ranking, relevant))


def test_recall_mrr_precision():
    relevant = {"a", "b", "c", "d"}
    ranking = ["x", "a", "y", "b"]
    assert recall_at_k(ranking, relevant, 2) == 0.25
    assert recall_at_k(ranking, relevant, 10) == 0.5
    assert recall_at_k(ranking, set(), 10) == 0.0
    assert mrr_at_k(ranking, relevant) == 0.5
    assert mrr_at_k(ranking, relevant, k=1) == 0.0
    assert precision_at_k(ranking, relevant, 4) == 0.5


def test_score_query_keys():
    scores = score_query(["a"], {"a": 1}, ndcg_k=10, recall_ks=(5, 100))
    assert set(scores) == {"ndcg@10", "mrr@10", "recall@5", "recall@100"}


def test_matches_trec_eval_on_random_runs():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(11)
    docs = [f"d{i}" for i in range(60)]
    qrels, run, rankings = {}, {}, {}
    for q in range(50):
        qid = f"q{q}"
        qrels[qid] = {d: rng.choice([1, 1, 2, 3]) for d in rng.sample(docs, rng.randint(1, 15))}
        ranking = rng.sample(docs, 30)
        rankings[qid] = ranking
        # trec_eval sorts by score, so give strictly decreasing scores.
        run[qid] = {d: float(len(ranking) - i) for i, d in enumerate(ranking)}
    measures = {"ndcg_cut.10", "recall.10", "recall.100", "recip_rank"}
    official = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    for qid, ranking in rankings.items():
        grades = qrels[qid]
        assert ndcg_at_k(ranking, grades, 10) == pytest.approx(
            official[qid]["ndcg_cut_10"], abs=1e-9
        )
        assert recall_at_k(ranking, grades, 10) == pytest.approx(
            official[qid]["recall_10"], abs=1e-9
        )
        assert recall_at_k(ranking, grades, 100) == pytest.approx(
            official[qid]["recall_100"], abs=1e-9
        )
        # trec_eval's recip_rank has no cutoff; with 30 results, compare at k=30.
        assert mrr_at_k(ranking, grades, 30) == pytest.approx(official[qid]["recip_rank"], abs=1e-9)
