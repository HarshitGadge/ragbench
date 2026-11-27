from pathlib import Path

import pytest

from ragbench.cli import main
from ragbench.gate import gate
from ragbench.retriever import load_retriever
from ragbench.retrievers import BM25
from ragbench.runner import RunResult, evaluate

OVERLAP = str(Path(__file__).parents[1] / "examples" / "overlap.py") + ":WordOverlap"


def WordOverlap(**params):
    return load_retriever(OVERLAP, params)


@pytest.fixture
def runs(tiny):
    base = evaluate(BM25(), tiny, name="bm25", bootstrap_rounds=200)
    same = evaluate(WordOverlap(), tiny, name="overlap", bootstrap_rounds=200)
    worse = evaluate(WordOverlap(min_overlap=2), tiny, name="overlap2", bootstrap_rounds=200)
    return base, same, worse


def test_unchanged_run_passes(runs):
    base, same, _ = runs
    result = gate(base, same, rounds=200)
    assert result.passed
    assert result.checks[0].comparison.diff.mean == pytest.approx(0.0)


def test_drop_beyond_threshold_fails(runs):
    base, _, worse = runs
    result = gate(base, worse, max_drop=0.01, rounds=200)
    assert not result.passed
    assert "FAIL ndcg@10" in result.checks[0].line()


def test_drop_within_threshold_passes(runs):
    base, _, worse = runs
    drop = -gate(base, worse, rounds=200).checks[0].comparison.diff.mean
    assert gate(base, worse, max_drop=drop + 0.001, rounds=200).passed
    assert not gate(base, worse, max_drop=drop - 0.001, rounds=200).passed


def test_ci_mode_uses_the_upper_end_of_the_interval(runs):
    base, _, worse = runs
    d = gate(base, worse, rounds=200).checks[0].comparison.diff
    # A threshold between the CI's upper end and the mean: mean mode fails, ci mode passes.
    threshold = (-d.mean + -d.high) / 2
    assert -d.high < threshold < -d.mean
    assert not gate(base, worse, max_drop=threshold, mode="mean", rounds=200).passed
    assert gate(base, worse, max_drop=threshold, mode="ci", rounds=200).passed


def test_every_metric_must_pass(runs):
    base, _, worse = runs
    result = gate(base, worse, metrics=["mrr@10", "ndcg@10"], rounds=200)
    by_metric = {c.comparison.metric: c.passed for c in result.checks}
    assert by_metric == {"mrr@10": True, "ndcg@10": False}
    assert not result.passed


def test_refuses_runs_with_different_settings(tiny):
    a = evaluate(BM25(), tiny, name="a", depth=100, bootstrap_rounds=50)
    b = evaluate(BM25(), tiny, name="b", depth=200, bootstrap_rounds=50)
    with pytest.raises(ValueError, match="depth"):
        gate(a, b)


def test_rejects_bad_mode_and_negative_drop(runs):
    base, same, _ = runs
    with pytest.raises(ValueError, match="mode"):
        gate(base, same, mode="median")
    with pytest.raises(ValueError, match="negative"):
        gate(base, same, max_drop=-0.1)


def test_cli_exit_codes_and_summary(runs, tmp_path, capsys, monkeypatch):
    base, same, worse = runs
    for run in runs:
        run.save(tmp_path / f"{run.name}.json")
    summary = tmp_path / "summary.md"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    ok = main(
        [
            "gate",
            str(tmp_path / "bm25.json"),
            str(tmp_path / "overlap.json"),
            "--rounds",
            "200",
            "--summary",
            str(summary),
        ]
    )
    assert ok == 0
    assert "### ragbench gate passed on tiny" in summary.read_text()

    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bad = main(
        ["gate", str(tmp_path / "bm25.json"), str(tmp_path / "overlap2.json"), "--rounds", "200"]
    )
    assert bad == 1
    assert "### ragbench gate failed on tiny" in summary.read_text()
    assert "FAIL ndcg@10" in capsys.readouterr().out


def test_committed_baseline_matches_a_fresh_run(tiny):
    """tests/fixtures/gate/tiny-bm25.json is the baseline the action's self-test uses."""
    path = Path(__file__).parent / "fixtures" / "gate" / "tiny-bm25.json"
    committed = RunResult.load(path)
    fresh = evaluate(BM25(), tiny, name="bm25")
    assert committed.per_query == fresh.per_query
