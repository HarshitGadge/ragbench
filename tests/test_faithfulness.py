import itertools
import json
import random

import pytest

from ragbench.cli import main
from ragbench.faithfulness import (
    AnswerRecord,
    load_answers,
    load_judge,
    roc_auc,
    score_answers,
    split_claims,
)
from ragbench.judges import TokenOverlap, content_words

from .conftest import ROOT

ANSWERS = ROOT / "tests" / "fixtures" / "faithfulness" / "answers.jsonl"


@pytest.mark.parametrize(
    ("answer", "claims"),
    [
        ("One claim. Two claims! Three?", ["One claim.", "Two claims!", "Three?"]),
        ("Rates rose 3.5 percent. Then fell.", ["Rates rose 3.5 percent.", "Then fell."]),
        (
            "Fees apply, e.g. Late fees. The U.S. Treasury agrees.",
            ["Fees apply, e.g. Late fees.", "The U.S. Treasury agrees."],
        ),
        (
            "- first point\n- second point\n\n1. third point",
            ["first point", "second point", "third point"],
        ),
        (
            "See 12 CFR 1005.6(b). It limits liability.",
            ["See 12 CFR 1005.6(b).", "It limits liability."],
        ),
        ('He said "Stop." Then left.', ['He said "Stop."', "Then left."]),
        ("", []),
        ("Ok", []),
    ],
)
def test_split_claims(answer, claims):
    assert split_claims(answer) == claims


def test_token_overlap_scores_best_context():
    judge = TokenOverlap()
    contexts = ["notice within 60 days", "liability limited to fifty dollars"]
    assert judge.support("Notice within 60 days.", contexts) == pytest.approx(1.0)
    # "refund" and "losses" are missing from both contexts; "notice" is in the first.
    assert judge.support("Notice brings a refund of losses.", contexts) == pytest.approx(1 / 4)
    assert judge.support("It is so.", contexts) == 1.0  # only stopwords: nothing to check
    assert judge.support("Anything at all.", []) == 0.0


def test_negation_counts_as_content():
    assert "not" in content_words("does not apply")
    judge = TokenOverlap()
    assert judge.support("The rule does not apply.", ["The rule applies."]) < 1.0


def test_roc_auc_known_values():
    assert roc_auc([0.9, 0.8, 0.4, 0.3], [1, 0, 1, 0]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert roc_auc([1.0, 0.0], [1, 0]) == 1.0
    assert roc_auc([0.0, 1.0], [1, 0]) == 0.0
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2], [1, 1])


def test_roc_auc_matches_pairwise_count():
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randint(2, 30)
        scores = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        if len(set(labels)) < 2:
            continue
        pos = [s for s, y in zip(scores, labels, strict=True) if y]
        neg = [s for s, y in zip(scores, labels, strict=True) if not y]
        wins = sum((p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg))
        assert roc_auc(scores, labels) == pytest.approx(wins / (len(pos) * len(neg)))


def test_score_answers_on_fixture():
    records = load_answers(ANSWERS)
    result = score_answers(
        TokenOverlap(), records, judge_spec="ragbench.judges:TokenOverlap", rounds=200
    )
    by_id = {a["id"]: a for a in result.answers}
    assert by_id["a1"]["faithfulness"] == 1.0
    assert by_id["a2"]["faithfulness"] == 0.5  # first claim supported, the refund claim not
    assert by_id["a3"]["faithfulness"] == 1.0
    assert by_id["a4"]["faithfulness"] == 0.0
    # A faithful paraphrase ("CFPB" for "Bureau") shares no words with its context, so
    # the lexical baseline calls it unsupported: the kind of miss a real judge must fix.
    assert by_id["a5"]["faithfulness"] == 0.0
    s = result.summary
    assert (s["answers"], s["scored"], s["no_claims"]) == (5, 5, 0)
    assert s["faithfulness"]["mean"] == pytest.approx((1 + 0.5 + 1 + 0 + 0) / 5)
    low, high = s["faithfulness"]["ci"]
    assert low <= s["faithfulness"]["mean"] <= high
    a = result.agreement
    assert a["labelled"] == 5 and a["faithful"] == 3
    # Faithful answers score 1, 1, 0 and unfaithful ones 0.5, 0: 4.5 of 6 pairs ordered right.
    assert a["auc_faithfulness"] == pytest.approx(4.5 / 6)
    assert a["accuracy_all_supported"] == pytest.approx(4 / 5)
    assert result.name == "TokenOverlap"


def test_threshold_changes_which_claims_count():
    rec = AnswerRecord("x", "", ["alpha beta"], "Alpha beta gamma delta.")
    loose = score_answers(TokenOverlap(), [rec], threshold=0.5, rounds=50)
    strict = score_answers(TokenOverlap(), [rec], threshold=0.6, rounds=50)
    assert loose.answers[0]["faithfulness"] == 1.0
    assert strict.answers[0]["faithfulness"] == 0.0
    with pytest.raises(ValueError):
        score_answers(TokenOverlap(), [rec], threshold=1.5)


def test_answers_without_claims_are_left_out():
    recs = [AnswerRecord("a", "", ["x y"], "x y z."), AnswerRecord("b", "", ["x"], "")]
    result = score_answers(TokenOverlap(), recs, rounds=50)
    assert result.summary["no_claims"] == 1
    assert result.answers[1]["faithfulness"] is None
    with pytest.raises(ValueError, match="no answer had a claim"):
        score_answers(TokenOverlap(), [recs[1]])


class _Batch:
    def __init__(self):
        self.calls = 0

    def support_batch(self, items):
        self.calls += 1
        return [0.9 if "good" in claim else 0.1 for claim, _, _ in items]


def test_support_batch_is_used_once_per_answer():
    judge = _Batch()
    rec = AnswerRecord("a", "q", ["ctx"], "A good claim. A bad claim.")
    result = score_answers(judge, [rec], rounds=50)
    assert judge.calls == 1
    assert result.answers[0]["faithfulness"] == 0.5


def test_judge_scores_must_be_probabilities():
    class Bad:
        def support(self, claim, contexts, question):
            return 2.0

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        score_answers(Bad(), [AnswerRecord("a", "", ["c"], "A claim here.")])


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"id": "a", "answer": "x"}, "missing contexts"),
        ({"id": "a", "answer": "x", "contexts": "c"}, "list of strings"),
        ({"id": "a", "answer": "x", "contexts": [], "label": 2}, "label must be 0 or 1"),
    ],
)
def test_load_answers_rejects_bad_rows(tmp_path, line, message):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(line) + "\n")
    with pytest.raises(ValueError, match=message):
        load_answers(path)


def test_load_answers_rejects_duplicates_and_empty(tmp_path):
    path = tmp_path / "a.jsonl"
    row = {"id": "a", "answer": "x", "contexts": []}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_answers(path)
    path.write_text("\n")
    with pytest.raises(ValueError, match="no answers"):
        load_answers(path)


def test_load_judge(tmp_path):
    assert isinstance(load_judge("ragbench.judges:TokenOverlap", {"min_words": 2}), TokenOverlap)
    plugin = tmp_path / "myjudge.py"
    plugin.write_text(
        "class Always:\n    def support(self, claim, contexts, question):\n        return 1.0\n"
    )
    assert load_judge(f"{plugin}:Always").support("x", [], "") == 1.0
    with pytest.raises(TypeError, match="no support"):
        load_judge("ragbench.judges:_WORD")


def test_cli_faithfulness(tmp_path, capsys):
    out = tmp_path / "f.json"
    code = main(
        ["faithfulness", "--answers", str(ANSWERS), "--out", str(out), "--rounds", "200", "--quiet"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "faithfulness 0.5000" in printed
    assert "agreement with 5 human labels: ROC AUC 0.7500" in printed
    data = json.loads(out.read_text())
    assert data["schema"] == "ragbench.faithfulness/1"
    assert data["judge"] == "ragbench.judges:TokenOverlap"
    assert len(data["answers"]) == 5
    assert data["meta"]["command"].startswith("ragbench faithfulness --answers")


def test_cli_faithfulness_bad_judge(capsys):
    assert main(["faithfulness", "--answers", str(ANSWERS), "--judge", "ragbench.judges:Nope"]) == 2
    assert "error" in capsys.readouterr().err
