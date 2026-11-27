import json

import pytest

from ragbench import Hit, evaluate
from ragbench.compare import compare_runs
from ragbench.retriever import doc_ids, load_retriever, parse_params
from ragbench.runner import RunResult


class Oracle:
    """Returns the judged documents, best grade first: every metric should be 1."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.by_text = {t: q for q, t in dataset.queries.items()}

    def search(self, query, k):
        grades = self.dataset.qrels[self.by_text[query]]
        return [Hit(d, g) for d, g in sorted(grades.items(), key=lambda x: -x[1])][:k]


class Nothing:
    def search(self, query, k):
        return []


class Chunky:
    """Returns chunk-level duplicates; each document must count once."""

    indexed = None

    def index(self, corpus):
        self.indexed = len(corpus)

    def search(self, query, k):
        return [("d1", 0.9), ("d1", 0.8), ("d2", 0.7)]


class Batched:
    def __init__(self):
        self.calls = 0

    def search(self, query, k):  # pragma: no cover - search_batch wins
        raise AssertionError("search_batch should be used")

    def search_batch(self, queries, k):
        self.calls += 1
        return [["d1"] for _ in queries]


def test_oracle_scores_one(tiny):
    result = evaluate(Oracle(tiny), tiny, name="oracle", bootstrap_rounds=100)
    for metric in ("ndcg@10", "recall@10", "recall@100", "mrr@10"):
        assert result.mean(metric) == pytest.approx(1.0)
    assert result.dataset["evaluated_queries"] == 3


def test_empty_retriever_scores_zero(tiny):
    result = evaluate(Nothing(), tiny, bootstrap_rounds=100)
    assert result.mean("ndcg@10") == 0.0


def test_chunk_duplicates_count_once_and_index_called(tiny):
    r = Chunky()
    result = evaluate(r, tiny, bootstrap_rounds=100)
    assert r.indexed == 6
    # q1: d1 (2) at rank 1, d2 (1) at rank 2 -> ideal ordering
    assert result.per_query["ndcg@10"]["q1"] == pytest.approx(1.0)
    assert result.per_query["recall@10"]["q2"] == 0.0


def test_search_batch_is_used(tiny):
    r = Batched()
    result = evaluate(r, tiny, bootstrap_rounds=100)
    assert r.calls == 1
    assert result.latency_ms["batched"] is True


def test_limit_and_depth_validation(tiny):
    result = evaluate(Nothing(), tiny, limit=2, bootstrap_rounds=50)
    assert result.dataset["evaluated_queries"] == 2
    with pytest.raises(ValueError, match="shallower"):
        evaluate(Nothing(), tiny, depth=5)


def test_result_round_trip(tiny, tmp_path):
    result = evaluate(Oracle(tiny), tiny, name="oracle", bootstrap_rounds=100)
    path = result.save(tmp_path / "r.json")
    data = json.loads(path.read_text())
    assert data["metrics"]["ndcg@10"]["mean"] == 1.0
    assert data["meta"]["ragbench"]
    loaded = RunResult.load(path)
    assert loaded.per_query == result.per_query


def test_compare_runs(tiny):
    good = evaluate(Oracle(tiny), tiny, name="oracle", bootstrap_rounds=100)
    bad = evaluate(Nothing(), tiny, name="nothing", bootstrap_rounds=100)
    c = compare_runs(bad, good, "ndcg@10", rounds=200)
    assert c.diff.mean == pytest.approx(1.0)
    assert c.significant
    with pytest.raises(KeyError):
        compare_runs(bad, good, "ndcg@5")


def test_compare_rejects_mismatched_queries(tiny):
    a = evaluate(Nothing(), tiny, bootstrap_rounds=50)
    b = evaluate(Nothing(), tiny, limit=2, bootstrap_rounds=50)
    with pytest.raises(ValueError, match="different queries"):
        compare_runs(a, b)


def test_doc_ids_normalisation():
    class Obj:
        doc_id = "z"

    assert doc_ids(["a", Hit("b", 1.0), ("c", 0.5), Obj()]) == ["a", "b", "c", "z"]
    with pytest.raises(TypeError):
        doc_ids([3.14])


def test_parse_params():
    assert parse_params(["k1=0.9", "name=bm25", "flags=[1,2]", "on=true"]) == {
        "k1": 0.9,
        "name": "bm25",
        "flags": [1, 2],
        "on": True,
    }
    with pytest.raises(ValueError):
        parse_params(["oops"])


def test_load_retriever_from_file(tmp_path):
    path = tmp_path / "mine.py"
    path.write_text(
        "class R:\n"
        "    def __init__(self, n=1):\n"
        "        self.n = n\n"
        "    def search(self, q, k):\n"
        "        return []\n"
        "instance = R()\n"
        "class NoSearch:\n"
        "    pass\n"
    )
    assert load_retriever(f"{path}:R", {"n": 3}).n == 3
    assert load_retriever(f"{path}:instance").n == 1
    with pytest.raises(ValueError):
        load_retriever(f"{path}:instance", {"n": 2})
    with pytest.raises(TypeError):
        load_retriever(f"{path}:NoSearch")
    with pytest.raises(ValueError):
        load_retriever("no-colon")


def test_load_retriever_from_module():
    r = load_retriever("tests.test_runner:Nothing")
    assert r.search("q", 10) == []


class SelfFirst:
    """Retrieves a document with the query's own id first, as on ArguAna."""

    def __init__(self, dataset):
        self.by_text = {t: q for q, t in dataset.queries.items()}

    def search(self, query, k):
        return [self.by_text[query], "d1", "d4", "d5"]


def test_identical_ids_are_dropped_by_default(tiny):
    kept = evaluate(SelfFirst(tiny), tiny, bootstrap_rounds=50, ignore_identical_ids=False)
    dropped = evaluate(SelfFirst(tiny), tiny, bootstrap_rounds=50)
    # q1's relevant d1 moves from rank 2 to rank 1 once the self-match is removed
    assert dropped.per_query["mrr@10"]["q1"] == 1.0
    assert kept.per_query["mrr@10"]["q1"] == 0.5
    assert dropped.settings["ignore_identical_ids"] is True


def test_ci_run_recorded(tiny, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    result = evaluate(Nothing(), tiny, bootstrap_rounds=20)
    assert result.meta["ci_run"] == "https://github.com/o/r/actions/runs/42"
    assert result.meta["git_sha"] == "abc"


def test_evaluate_rejects_bad_recall_ks(tiny):
    for bad in [(), (0,), (5, 5), (2.5,)]:
        with pytest.raises(ValueError):
            evaluate(Oracle(tiny), tiny, recall_ks=bad, bootstrap_rounds=10)
    result = evaluate(Oracle(tiny), tiny, recall_ks=[5], bootstrap_rounds=10)
    assert result.settings["recall_ks"] == [5]
    assert result.mean("recall@5") == 1.0
