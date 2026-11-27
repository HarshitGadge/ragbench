import json

import pytest

from ragbench.cli import main
from ragbench.faithfulness import load_answers, merge_results, shard

from .conftest import ROOT

ANSWERS = ROOT / "tests" / "fixtures" / "faithfulness" / "answers.jsonl"


def test_shards_are_contiguous_and_cover_every_answer():
    records = load_answers(ANSWERS)
    for n in (1, 2, 3, 5, 8):
        parts = [shard(records, f"{k}/{n}") for k in range(n)]
        assert [r.id for p in parts for r in p] == [r.id for r in records]
        assert max(map(len, parts)) - min(map(len, parts)) <= 1


@pytest.mark.parametrize("spec", ["3/3", "-1/2", "1/0", "a/b", "2"])
def test_bad_shard_spec(spec):
    with pytest.raises(ValueError, match="K/N"):
        shard(load_answers(ANSWERS), spec)


def _run(tmp_path, *extra):
    out = tmp_path / ("all.json" if not extra else f"shard{extra[1].replace('/', 'of')}.json")
    argv = ["faithfulness", "--answers", str(ANSWERS), "--out", str(out), "--rounds", "300"]
    assert main([*argv, "--quiet", *extra]) == 0
    return out


def test_merged_shards_match_an_unsharded_run(tmp_path, capsys):
    whole = json.loads(_run(tmp_path).read_text())
    paths = [_run(tmp_path, "--shard", f"{k}/3") for k in (2, 0, 1)]  # any order
    merged_path = tmp_path / "merged.json"
    capsys.readouterr()
    assert (
        main(["faithfulness-merge", *map(str, paths), "--out", str(merged_path), "--rounds", "300"])
        == 0
    )
    assert "faithfulness 0.5000" in capsys.readouterr().out
    merged = json.loads(merged_path.read_text())
    for key in (
        "schema",
        "name",
        "judge",
        "params",
        "threshold",
        "summary",
        "agreement",
        "answers",
    ):
        assert merged[key] == whole[key], key
    assert merged["meta"]["merged_shards"] == 3
    assert len(merged["meta"]["shard_commands"]) == 3
    assert merged["meta"]["shard_commands"][0].endswith("--shard 0/3")


def test_merge_refuses_gaps_repeats_and_mixed_judges(tmp_path):
    a, b, c = (json.loads(_run(tmp_path, "--shard", f"{k}/3").read_text()) for k in range(3))
    with pytest.raises(ValueError, match="cover 0..2"):
        merge_results([a, b])
    with pytest.raises(ValueError, match="cover 0..2"):
        merge_results([a, a, c])
    other = dict(b, judge="ragbench.judges:NLI")
    with pytest.raises(ValueError, match="disagree on judge"):
        merge_results([a, other, c])
    with pytest.raises(ValueError, match="meta.shard"):
        merge_results([a, dict(b, meta={}), c])
