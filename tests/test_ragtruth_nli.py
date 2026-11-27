import json
from pathlib import Path

import pytest

from ragbench import ragtruth
from ragbench.faithfulness import load_answers, score_answers
from ragbench.judges import NLI, entailment_index

FIX = "tests/fixtures/ragtruth"


def test_split_passages():
    assert ragtruth.split_passages("passage 1:A. passage 2: B. ") == ["A.", "B."]


def test_convert_keeps_good_qa_test_answers(tmp_path):
    rows, counts = ragtruth.convert(f"{FIX}/response.jsonl", f"{FIX}/source_info.jsonl")
    assert [r["id"] for r in rows] == ["ragtruth-1", "ragtruth-2"]
    assert [r["label"] for r in rows] == [1, 0]
    assert rows[0]["contexts"] == ["The filing is due within 60 days.", "Late filings pay a fee."]
    assert rows[0]["question"] == "When is the filing due?"
    assert counts == {
        "answers": 2,
        "faithful": 1,
        "unfaithful": 1,
        "skipped": {"incorrect_refusal": 1},
    }
    # The rows are a valid answers file.
    path = tmp_path / "answers.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [a.label for a in load_answers(path)] == [1, 0]


def test_fetch_rejects_a_changed_file(tmp_path):
    (tmp_path / "response.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="does not match"):
        ragtruth._fetch("response.jsonl", tmp_path, timeout=1)


def test_build_writes_answers_and_about(tmp_path, monkeypatch):
    def fake_fetch(name, raw_dir, timeout):
        return Path(FIX) / name

    monkeypatch.setattr(ragtruth, "_fetch", fake_fetch)
    out = ragtruth.build(tmp_path)
    assert out == tmp_path / "ragtruth-qa" / "answers.jsonl"
    about = json.loads((out.parent / "about.json").read_text())
    assert about["commit"] == ragtruth.COMMIT
    assert about["answers"] == 2


def test_entailment_index():
    assert entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0
    assert entailment_index({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}) == 2
    with pytest.raises(ValueError):
        entailment_index({0: "LABEL_0", 1: "LABEL_1"})


class StubNLI(NLI):
    """Entailment = 0.9 when every claim word is in the premise, else 0.1."""

    def __init__(self):
        super().__init__(model="stub")
        self.calls = []

    def _predict(self, pairs):
        self.calls.append(list(pairs))
        return [0.9 if set(h.lower().split()) <= set(p.lower().split()) else 0.1 for p, h in pairs]


def test_nli_takes_the_best_context_per_claim():
    judge = StubNLI()
    items = [
        ("a b", ["x y", "a b c"], "q"),
        ("z", ["x y", "a b c"], "q"),
        ("a", [], "q"),
    ]
    assert judge.support_batch(items) == [0.9, 0.1, 0.0]
    # One model call for the whole batch, with the context as premise.
    assert len(judge.calls) == 1
    assert judge.calls[0][0] == ("x y", "a b")
    assert judge.support("c", ["a b c"]) == 0.9


def test_nli_no_pairs_skips_the_model():
    judge = StubNLI()
    assert judge.support_batch([("a", [], "")]) == [0.0]
    assert judge.calls == []


def test_nli_scores_answers_end_to_end():
    records = load_answers("tests/fixtures/faithfulness/answers.jsonl")
    result = score_answers(StubNLI(), records, judge_spec="stub", params={}, rounds=200)
    assert result.summary["answers"] == len(records)
    assert result.agreement["labelled"] == len(records)


def test_nli_is_lazy():
    # Building the judge loads nothing, so the CLI can validate it without the extra.
    judge = NLI(batch_size="8", max_length="256")
    assert judge._loaded is None
    assert (judge.batch_size, judge.max_length) == (8, 256)
