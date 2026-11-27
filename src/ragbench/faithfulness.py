"""Answer faithfulness: is every claim in a generated answer backed by the retrieved text?

Retrieval metrics say whether the right passages were found. They say nothing
about what the generator did with them. Faithfulness scores the answer: it is
split into claims (sentences), a *judge* rates how well the retrieved contexts
support each claim, and the answer's score is the share of claims the judge
calls supported. The run's score is the mean over answers, with a bootstrap
confidence interval, like every other RAGBench metric.

A judge is any object with ``support(claim, contexts, question) -> float`` that
returns a support score in [0, 1]. It may also have ``support_batch(items)``,
which takes a list of ``(claim, contexts, question)`` and returns a list of
scores. See :mod:`ragbench.judges` for the built-in judges.

Answer files are JSON Lines, one answer per line::

    {"id": "q1", "question": "...", "contexts": ["...", "..."], "answer": "..."}

A line may also carry ``"label": 1`` (faithful) or ``0`` (not faithful) from a
human. With labels, the run also reports how well the judge's scores agree
with the humans (ROC AUC, and accuracy at the threshold), which is how a judge
is checked before its scores are trusted.
"""

from __future__ import annotations

import inspect
import json
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragbench.retriever import _import_target
from ragbench.runner import environment
from ragbench.stats import bootstrap_ci

SCHEMA = "ragbench.faithfulness/1"

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "no", "nos", "sec", "secs", "art",
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "ltd", "co", "corp",
    "u.s", "u.k", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "approx", "p", "pp", "vol", "ch",
}  # fmt: skip
_BOUNDARY = re.compile(r"(?<=[.!?])([\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
_MIN_CLAIM_CHARS = 3


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims.

    Lines are split first (bullets and numbered lists are one claim per line),
    then sentences, without breaking after common abbreviations or inside
    numbers such as 3.5.
    """
    claims: list[str] = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            before = line[start : m.start()]
            last_word = before.rsplit(None, 1)[-1].lower().rstrip(".!?") if before else ""
            if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word):
                continue
            # Keep a closing quote or bracket with the sentence it closes.
            claims.append(line[start : m.start() + len(m.group(1))].strip())
            start = m.end()
        claims.append(line[start:].strip())
    return [c for c in claims if len(c) >= _MIN_CLAIM_CHARS]


@dataclass(frozen=True)
class AnswerRecord:
    id: str
    question: str
    contexts: list[str]
    answer: str
    label: int | None = None


def load_answers(path: str | Path) -> list[AnswerRecord]:
    records: list[AnswerRecord] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("id", "answer", "contexts") if k not in row]
            if missing:
                raise ValueError(f"{path}:{n}: missing {', '.join(missing)}")
            if not isinstance(row["contexts"], list):
                raise ValueError(f"{path}:{n}: contexts must be a list of strings")
            rid = str(row["id"])
            if rid in seen:
                raise ValueError(f"{path}:{n}: duplicate id {rid!r}")
            seen.add(rid)
            label = row.get("label")
            if label is not None and label not in (0, 1, True, False):
                raise ValueError(f"{path}:{n}: label must be 0 or 1, got {label!r}")
            records.append(
                AnswerRecord(
                    id=rid,
                    question=str(row.get("question", "")),
                    contexts=[str(c) for c in row["contexts"]],
                    answer=str(row["answer"]),
                    label=None if label is None else int(label),
                )
            )
    if not records:
        raise ValueError(f"{path}: no answers")
    return records


def load_judge(spec: str, params: Mapping[str, Any] | None = None) -> Any:
    """Build a judge from ``module:Name`` or ``path/to/file.py:Name``."""
    target = _import_target(spec)
    params = dict(params or {})
    if inspect.isclass(target) or inspect.isfunction(target):
        judge = target(**params)
    elif params:
        raise ValueError(f"{spec} is an instance, so it cannot take --param values")
    else:
        judge = target
    if not callable(getattr(judge, "support", None)) and not callable(
        getattr(judge, "support_batch", None)
    ):
        raise TypeError(
            f"{spec} built a {type(judge).__name__} with no support(claim, contexts, question)"
        )
    return judge


def _score_claims(judge: Any, items: list[tuple[str, list[str], str]]) -> list[float]:
    batch = getattr(judge, "support_batch", None)
    scores = list(batch(items)) if callable(batch) else [judge.support(*it) for it in items]
    if len(scores) != len(items):
        raise ValueError(f"judge returned {len(scores)} scores for {len(items)} claims")
    out = []
    for s in scores:
        s = float(s)
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"judge support scores must be in [0, 1], got {s}")
        out.append(s)
    return out


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve: the chance a random faithful answer outscores an unfaithful one.

    Ties count half. Equivalent to the Mann-Whitney U statistic over n_pos * n_neg.
    """
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        raise ValueError("roc_auc needs at least one faithful and one unfaithful label")
    ranked = sorted((s, i) for i, s in enumerate(list(pos) + list(neg)))
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(ranks[: len(pos)])
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


@dataclass
class FaithfulnessResult:
    name: str
    judge: str
    params: dict
    threshold: float
    answers: list[dict]
    summary: dict
    agreement: dict | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "judge": self.judge,
            "params": self.params,
            "threshold": self.threshold,
            "summary": self.summary,
            "agreement": self.agreement,
            "meta": self.meta,
            "answers": self.answers,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def score_answers(
    judge: Any,
    records: Iterable[AnswerRecord],
    *,
    judge_spec: str = "",
    params: Mapping[str, Any] | None = None,
    name: str = "",
    threshold: float = 0.5,
    rounds: int = 2000,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> FaithfulnessResult:
    """Score every answer with ``judge``; return per-answer and summary results."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    records = list(records)
    per_answer: list[dict] = []
    for n, rec in enumerate(records, 1):
        claims = split_claims(rec.answer)
        support = _score_claims(judge, [(c, rec.contexts, rec.question) for c in claims])
        supported = sum(s >= threshold for s in support)
        row: dict[str, Any] = {
            "id": rec.id,
            "faithfulness": round(supported / len(claims), 6) if claims else None,
            "mean_support": round(statistics.fmean(support), 6) if claims else None,
            "claims": [
                {"text": c, "support": round(s, 6)} for c, s in zip(claims, support, strict=True)
            ],
        }
        if rec.label is not None:
            row["label"] = rec.label
        per_answer.append(row)
        if progress:
            progress(n, len(records))

    scored = [r for r in per_answer if r["faithfulness"] is not None]
    if not scored:
        raise ValueError("no answer had a claim to score")
    values = [r["faithfulness"] for r in scored]
    summary = {
        "answers": len(per_answer),
        "scored": len(scored),
        "no_claims": len(per_answer) - len(scored),
        "claims": sum(len(r["claims"]) for r in scored),
        "faithfulness": bootstrap_ci(values, rounds=rounds, seed=seed).as_dict(),
        "mean_support": round(statistics.fmean(r["mean_support"] for r in scored), 4),
    }

    agreement = None
    labelled = [r for r in scored if "label" in r]
    if labelled and {r["label"] for r in labelled} == {0, 1}:
        labels = [r["label"] for r in labelled]
        agreement = {
            "labelled": len(labelled),
            "faithful": sum(labels),
            "auc_faithfulness": round(roc_auc([r["faithfulness"] for r in labelled], labels), 4),
            "auc_mean_support": round(roc_auc([r["mean_support"] for r in labelled], labels), 4),
            # An answer is called faithful when every claim is supported.
            "accuracy_all_supported": round(
                statistics.fmean(
                    (r["faithfulness"] == 1.0) == bool(y)
                    for r, y in zip(labelled, labels, strict=True)
                ),
                4,
            ),
        }

    return FaithfulnessResult(
        name=name or judge_spec.rpartition(":")[2] or type(judge).__name__,
        judge=judge_spec,
        params=dict(params or {}),
        threshold=threshold,
        answers=per_answer,
        summary=summary,
        agreement=agreement,
        meta=environment() | dict(meta or {}),
    )
