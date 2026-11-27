import importlib.util
import json

from ragbench.faithfulness import load_answers, score_answers
from ragbench.judges import TokenOverlap

from .conftest import ROOT
from .test_update_readme import write_results

spec = importlib.util.spec_from_file_location("leaderboard", ROOT / "scripts" / "leaderboard.py")
leaderboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaderboard)


def test_build_and_check(tiny, tmp_path):
    results = tmp_path / "results"
    write_results(tiny, results)
    records = load_answers(ROOT / "tests/fixtures/faithfulness/answers.jsonl")
    score_answers(TokenOverlap(), records, name="token-overlap", rounds=100).save(
        results / "faithfulness" / "fixture" / "token-overlap.json"
    )
    (results / "faithfulness" / "fixture" / "about.json").write_text("{}")
    out = tmp_path / "leaderboard.json"
    args = ["--results", str(results), "--out", str(out)]
    assert leaderboard.main([*args, "--check"]) == 1
    assert leaderboard.main(args) == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "ragbench.leaderboard/1"
    # Best nDCG@10 first within a dataset, and no per-query scores.
    assert [r["run"] for r in data["retrieval"]] == ["oracle", "nothing"]
    assert "per_query" not in out.read_text()
    (comp,) = data["comparisons"]
    assert comp["candidate"] == "oracle"
    assert comp["source"] == next(r["source"] for r in data["retrieval"] if r["run"] == "oracle")
    prefix = "https://github.com/HarshitGadge/ragbench/blob/main/results/"
    assert comp["file"] == prefix + "tiny/compare.json"
    assert all(r["file"].startswith(prefix) for r in data["retrieval"] + data["faithfulness"])
    (f,) = data["faithfulness"]
    assert (f["answers"], f["judge"], f["agreement"]["auc_faithfulness"]) == (
        "fixture",
        "token-overlap",
        0.75,
    )
    assert leaderboard.main([*args, "--check"]) == 0


def test_committed_leaderboard_is_current():
    assert leaderboard.main(["--check"]) == 0
