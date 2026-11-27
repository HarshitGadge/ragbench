"""Download a BEIR dataset from the BEIR project's own mirror (stdlib only)."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# Public BEIR datasets that ship with a test split in the standard layout. The big
# ones (msmarco, climate-fever, ...) work too; these are the ones worth starting with.
KNOWN = {
    "scifact": "5,183 scientific abstracts, 300 test claims",
    "nfcorpus": "3,633 medical documents, 323 test queries, graded judgements",
    "arguana": "8,674 arguments, 1,406 test queries (counter-argument retrieval)",
    "fiqa": "57,638 financial forum posts, 648 test questions",
    "scidocs": "25,657 papers, 1,000 test queries",
    "trec-covid": "171,332 COVID-19 papers, 50 test queries, graded judgements",
}


def download(name: str, dest: str | Path = "data", url: str | None = None) -> Path:
    """Fetch ``<name>.zip`` and unpack it to ``dest/<name>``. Returns that folder."""
    dest = Path(dest)
    target = dest / name
    if (target / "corpus.jsonl").exists():
        return target
    source = url or BEIR_URL.format(name=name)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ragbench-") as tmp:
        archive = Path(tmp) / f"{name}.zip"
        with urllib.request.urlopen(source, timeout=300) as response, archive.open("wb") as fh:  # noqa: S310
            shutil.copyfileobj(response, fh)
        with zipfile.ZipFile(archive) as zf:
            root = Path(tmp) / "unzipped"
            for member in zf.namelist():
                resolved = (root / member).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(root)
        found = next(root.rglob("corpus.jsonl"), None)
        if found is None:
            raise ValueError(f"{source} has no corpus.jsonl")
        shutil.move(str(found.parent), str(target))
    return target
