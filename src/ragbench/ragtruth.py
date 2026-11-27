"""RAGTruth: human-labelled answers for checking a faithfulness judge.

RAGTruth (Niu et al., ACL 2024; https://github.com/ParticleMedia/RAGTruth, MIT
licence) has about 18,000 answers written by six LLMs from retrieved passages.
Annotators marked every span the passages do not support. This module turns
its question-answering test split into a RAGBench answers file: an answer is
labelled faithful (1) when annotators marked no span, else unfaithful (0).

The files are fetched from a pinned commit and checked against their SHA-256,
so every run scores the same answers.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
BASE = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{COMMIT}/dataset"
FILES = {
    "response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
}
NAME = "ragtruth-qa"
ABOUT = "875 LLM answers with human faithfulness labels (for checking a judge)"
_PASSAGE = re.compile(r"passage \d+:", re.IGNORECASE)


def split_passages(passages: str) -> list[str]:
    """RAGTruth joins the retrieved passages as "passage 1:... passage 2:..."."""
    return [p.strip() for p in _PASSAGE.split(passages) if p.strip()]


def _fetch(name: str, raw_dir: Path, timeout: float) -> Path:
    path = raw_dir / name
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=timeout) as response:  # noqa: S310
            data = response.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FILES[name]:
        raise ValueError(f"{path}: sha256 {digest} does not match the pinned RAGTruth file")
    return path


def convert(response_path: Path, source_path: Path, split: str = "test") -> tuple[list[dict], dict]:
    """Build answer rows for RAGTruth's QA task. Returns (rows, counts)."""
    sources = {}
    with open(source_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            sources[row["source_id"]] = row
    rows: list[dict] = []
    skipped: Counter[str] = Counter()
    with open(response_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            source = sources[r["source_id"]]
            if source["task_type"] != "QA" or r["split"] != split:
                continue
            if r.get("quality", "good") != "good":
                # Refusals and truncated answers say nothing about grounding.
                skipped[r["quality"]] += 1
                continue
            info = source["source_info"]
            rows.append(
                {
                    "id": f"ragtruth-{r['id']}",
                    "question": info["question"],
                    "contexts": split_passages(info["passages"]),
                    "answer": r["response"],
                    "label": 0 if r["labels"] else 1,
                    "model": r["model"],
                }
            )
    counts = {
        "answers": len(rows),
        "faithful": sum(r["label"] for r in rows),
        "unfaithful": sum(1 - r["label"] for r in rows),
        "skipped": dict(sorted(skipped.items())),
    }
    return rows, counts


def build(dest: str | Path = "data", timeout: float = 300) -> Path:
    """Download the pinned RAGTruth files and write ``dest/ragtruth-qa/answers.jsonl``."""
    dest = Path(dest)
    raw = dest / "ragtruth-raw"
    rows, counts = convert(
        _fetch("response.jsonl", raw, timeout), _fetch("source_info.jsonl", raw, timeout)
    )
    out = dest / NAME / "answers.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"source": BASE, "commit": COMMIT, "split": "test", "task": "QA", **counts}
    (out.parent / "about.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out
