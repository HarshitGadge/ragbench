import importlib.util
import json

from ragbench import evaluate

from .conftest import ROOT
from .test_runner import Nothing, Oracle

spec = importlib.util.spec_from_file_location(
    "update_readme", ROOT / "scripts" / "update_readme.py"
)
update_readme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_readme)


def write_results(tiny, folder):
    good = evaluate(Oracle(tiny), tiny, name="oracle", bootstrap_rounds=100)
    bad = evaluate(Nothing(), tiny, name="nothing", bootstrap_rounds=100)
    good.save(folder / "tiny" / "oracle.json")
    bad.save(folder / "tiny" / "nothing.json")
    from ragbench.compare import compare_runs

    row = compare_runs(bad, good, "ndcg@10", rounds=100).as_dict()
    (folder / "tiny" / "compare.json").write_text(json.dumps([row]))


def test_render_and_check(tiny, tmp_path):
    results = tmp_path / "results"
    write_results(tiny, results)
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:start -->\nstale\n<!-- results:end -->\noutro\n")
    args = ["--readme", str(readme), "--results", str(results)]
    assert update_readme.main([*args, "--check"]) == 1
    assert update_readme.main(args) == 0
    text = readme.read_text()
    assert "| tiny | nothing |" in text and "| tiny | oracle |" in text
    assert "tiny, ndcg@10: +1.0000" in text
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert update_readme.main([*args, "--check"]) == 0


def test_empty_results_and_missing_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start --><!-- results:end -->")
    assert update_readme.main(["--readme", str(readme), "--results", str(tmp_path)]) == 0
    assert "No results yet" in readme.read_text()
    readme.write_text("no markers")
    assert update_readme.main(["--readme", str(readme), "--results", str(tmp_path)]) == 2


def test_one_block_per_compared_pair(tiny, tmp_path):
    from ragbench.compare import compare_runs

    results = tmp_path / "results"
    write_results(tiny, results)
    good = evaluate(Oracle(tiny), tiny, name="oracle", bootstrap_rounds=100)
    bad = evaluate(Nothing(), tiny, name="nothing", bootstrap_rounds=100)
    row = compare_runs(good, bad, "ndcg@10", rounds=100).as_dict()
    (results / "tiny" / "compare-oracle__nothing.json").write_text(json.dumps([row]))
    text = update_readme.render(results)
    assert "Paired bootstrap, oracle minus nothing, same queries:" in text
    assert "Paired bootstrap, nothing minus oracle, same queries:" in text
    assert "| tiny | oracle |" in text and "compare" not in text.split("Means")[0]


def test_faithfulness_block(tmp_path):
    from ragbench.faithfulness import load_answers, score_answers
    from ragbench.judges import TokenOverlap

    records = load_answers(ROOT / "tests/fixtures/faithfulness/answers.jsonl")
    result = score_answers(TokenOverlap(), records, name="token-overlap", rounds=100)
    folder = tmp_path / "results" / "faithfulness" / "fixture"
    result.save(folder / "token-overlap.json")
    (folder / "about.json").write_text("{}")
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- results:start --><!-- results:end -->\n"
        "<!-- faithfulness:start -->old<!-- faithfulness:end -->\n"
    )
    args = ["--readme", str(readme), "--results", str(tmp_path / "results")]
    assert update_readme.main([*args, "--check"]) == 1
    assert update_readme.main(args) == 0
    text = readme.read_text()
    assert "| fixture (5) | token-overlap | 0.500 (" in text
    assert "| 0.750 | 0.800 |" in text
    assert "old" not in text
    assert update_readme.main([*args, "--check"]) == 0


def test_faithfulness_block_is_optional(tmp_path):
    assert update_readme.render_faithfulness(tmp_path) == "No faithfulness results yet.\n"
