"""Download a BEIR dataset.

Two sources, tried in order with ``source="auto"``:

* ``ukp``: the BEIR project's own zip mirror at TU Darmstadt (standard library only).
* ``hf``: the BEIR copies on the Hugging Face Hub (``BeIR/<name>`` and
  ``BeIR/<name>-qrels``), converted to the standard layout. Needs ``pyarrow``
  (``pip install 'ragbench-eval[hf]'``).

Either way the result is ``dest/<name>/{corpus.jsonl, queries.jsonl, qrels/<split>.tsv}``.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
HF = "https://huggingface.co"

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


def _fetch(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _from_zip(name: str, target: Path, url: str, timeout: float) -> None:
    with tempfile.TemporaryDirectory(prefix="ragbench-") as tmp:
        archive = Path(tmp) / f"{name}.zip"
        with urllib.request.urlopen(url, timeout=timeout) as response, archive.open("wb") as fh:  # noqa: S310
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
            raise ValueError(f"{url} has no corpus.jsonl")
        shutil.move(str(found.parent), str(target))


def _hf_files(repo: str, folder: str, timeout: float) -> list[str]:
    listing = json.loads(_fetch(f"{HF}/api/datasets/{repo}/tree/main/{folder}", timeout))
    files = sorted(f["path"] for f in listing if f.get("type") == "file")
    if not files:
        raise ValueError(f"no files under {repo}/{folder}")
    return files


def _from_hf(name: str, target: Path, split: str, timeout: float) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "downloading from Hugging Face needs: pip install 'ragbench-eval[hf]'"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="ragbench-") as tmp:
        staging = Path(tmp) / name
        (staging / "qrels").mkdir(parents=True)
        for part in ("corpus", "queries"):
            with (staging / f"{part}.jsonl").open("w", encoding="utf-8") as out:
                for path in _hf_files(f"BeIR/{name}", part, timeout):
                    if not path.endswith(".parquet"):
                        continue
                    data = _fetch(f"{HF}/datasets/BeIR/{name}/resolve/main/{path}", timeout)
                    for row in pq.read_table(io.BytesIO(data)).to_pylist():
                        out.write(json.dumps(row) + "\n")
        qrels = _fetch(f"{HF}/datasets/BeIR/{name}-qrels/resolve/main/{split}.tsv", timeout)
        (staging / "qrels" / f"{split}.tsv").write_bytes(qrels)
        shutil.move(str(staging), str(target))


def download(
    name: str,
    dest: str | Path = "data",
    url: str | None = None,
    source: str = "auto",
    split: str = "test",
    timeout: float = 60,
) -> Path:
    """Fetch a BEIR dataset into ``dest/<name>`` and return that folder.

    ``url`` overrides the zip location (and implies ``source="ukp"``).
    """
    dest = Path(dest)
    target = dest / name
    if (target / "corpus.jsonl").exists():
        return target
    if source not in ("auto", "ukp", "hf"):
        raise ValueError(f"source must be auto, ukp or hf, not {source!r}")
    dest.mkdir(parents=True, exist_ok=True)

    if url or source in ("auto", "ukp"):
        try:
            _from_zip(name, target, url or BEIR_URL.format(name=name), timeout)
            return target
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if url or source == "ukp":
                raise
            print(f"BEIR mirror unavailable ({exc}); trying Hugging Face", file=sys.stderr)
    _from_hf(name, target, split, timeout)
    return target
