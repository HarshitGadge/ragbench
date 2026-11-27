"""BEIR-format datasets: ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/<split>.tsv``.

The layout is the one the BEIR benchmark publishes (Thakur et al., 2021), so any
BEIR dataset, or your own data written the same way, loads without conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One corpus entry. ``text`` is what a retriever should index."""

    id: str
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip() if self.title else self.body.strip()


@dataclass
class Dataset:
    """A corpus, its queries and graded relevance judgements.

    ``qrels[query_id][doc_id]`` is the relevance grade. Grades of 0 or below mean
    "judged not relevant" and are dropped at load time, as BEIR's own evaluation does.
    Only queries with at least one relevant document are kept.
    """

    name: str
    corpus: dict[str, Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    split: str = "test"
    meta: dict = field(default_factory=dict)

    def relevant(self, query_id: str) -> set[str]:
        return set(self.qrels.get(query_id, {}))

    def summary(self) -> dict:
        judgements = sum(len(v) for v in self.qrels.values())
        return {
            "name": self.name,
            "split": self.split,
            "documents": len(self.corpus),
            "queries": len(self.queries),
            "judgements": judgements,
        }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read a BEIR qrels TSV (``query-id  corpus-id  score``, header row optional)."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            query_id, doc_id, score = parts[0], parts[1], parts[2]
            try:
                grade = int(float(score))
            except ValueError:
                if line_no == 1:  # header row
                    continue
                raise ValueError(f"{path}:{line_no}: score {score!r} is not a number") from None
            if grade > 0:
                qrels.setdefault(query_id, {})[doc_id] = grade
    return qrels


def load_beir(folder: str | Path, split: str = "test") -> Dataset:
    """Load a BEIR-format dataset folder."""
    folder = Path(folder)
    for required in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"):
        if not (folder / required).exists():
            raise FileNotFoundError(f"{folder / required} not found (expected BEIR layout)")

    corpus = {
        str(d["_id"]): Document(str(d["_id"]), d.get("title") or "", d.get("text") or "")
        for d in _read_jsonl(folder / "corpus.jsonl")
    }
    all_queries = {str(q["_id"]): q["text"] for q in _read_jsonl(folder / "queries.jsonl")}
    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")

    missing = sorted(qid for qid in qrels if qid not in all_queries)
    if missing:
        raise ValueError(
            f"{len(missing)} judged queries are missing from queries.jsonl: {missing[:5]}"
        )

    queries = {qid: all_queries[qid] for qid in sorted(qrels)}
    return Dataset(folder.name, corpus, queries, qrels, split=split)
