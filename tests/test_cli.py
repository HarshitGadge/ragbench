import json
import subprocess
import sys
import zipfile

from ragbench.cli import main
from ragbench.download import download

from .conftest import ROOT

OVERLAP = str(ROOT / "examples" / "overlap.py") + ":WordOverlap"


def test_run_report_compare(tiny_path, tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert (
        main(
            [
                "run",
                "--data",
                str(tiny_path),
                "--retriever",
                OVERLAP,
                "--out",
                str(a),
                "--quiet",
                "--rounds",
                "200",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "| tiny | WordOverlap |" in out
    assert (
        main(
            [
                "run",
                "--data",
                str(tiny_path),
                "--retriever",
                OVERLAP,
                "--param",
                "min_overlap=2",
                "--name",
                "strict",
                "--out",
                str(b),
                "--quiet",
                "--rounds",
                "200",
            ]
        )
        == 0
    )
    data = json.loads(a.read_text())
    assert data["meta"]["command"].startswith("ragbench run --data")
    assert data["meta"]["params"] == {}
    assert json.loads(b.read_text())["meta"]["params"] == {"min_overlap": 2}
    capsys.readouterr()

    assert main(["report", str(a), str(b)]) == 0
    table = capsys.readouterr().out
    assert "WordOverlap" in table and "strict" in table

    assert main(["compare", str(a), str(b), "--json", "--rounds", "200"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["metric"] == "ndcg@10"
    assert main(["compare", str(a), str(b), "--metric", "recall@10", "--rounds", "200"]) == 0
    assert "recall@10:" in capsys.readouterr().out


def test_overlap_example_scores(tiny_path, tmp_path):
    out = tmp_path / "r.json"
    main(
        [
            "run",
            "--data",
            str(tiny_path),
            "--retriever",
            OVERLAP,
            "--out",
            str(out),
            "--quiet",
            "--rounds",
            "100",
        ]
    )
    per_query = json.loads(out.read_text())["per_query"]["recall@10"]
    # every tiny query shares a word with its relevant documents
    assert per_query["q2"] == 1.0


def test_info_and_errors(tiny_path, capsys):
    assert main(["info"]) == 0
    assert "scifact" in capsys.readouterr().out
    assert main(["info", str(tiny_path)]) == 0
    assert json.loads(capsys.readouterr().out)["queries"] == 3
    assert main(["run", "--data", "/nonexistent", "--retriever", OVERLAP]) == 2
    assert "error" in capsys.readouterr().err


def test_download_from_local_zip(tiny_path, tmp_path):
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in tiny_path.rglob("*"):
            if f.is_file():
                zf.write(f, "tiny/" + str(f.relative_to(tiny_path)))
    folder = download("tiny", tmp_path / "data", url=archive.as_uri())
    assert (folder / "corpus.jsonl").exists()
    # second call is a no-op
    assert download("tiny", tmp_path / "data", url="file:///does-not-exist") == folder


def test_module_entry_point():
    proc = subprocess.run(
        [sys.executable, "-m", "ragbench", "--version"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("ragbench ")


def test_download_falls_back_to_hugging_face(tmp_path, monkeypatch):
    import io

    import pytest

    pa = pytest.importorskip("pyarrow")
    import urllib.error

    import pyarrow.parquet as pq

    from ragbench import download as dl

    def parquet(rows):
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buf)
        return buf.getvalue()

    files = {
        "/api/datasets/BeIR/toy/tree/main/corpus": json.dumps(
            [{"type": "file", "path": "corpus/corpus-00000-of-00001.parquet"}]
        ).encode(),
        "/api/datasets/BeIR/toy/tree/main/queries": json.dumps(
            [{"type": "file", "path": "queries/queries-00000-of-00001.parquet"}]
        ).encode(),
        "/datasets/BeIR/toy/resolve/main/corpus/corpus-00000-of-00001.parquet": parquet(
            [{"_id": "d1", "title": "T", "text": "body"}]
        ),
        "/datasets/BeIR/toy/resolve/main/queries/queries-00000-of-00001.parquet": parquet(
            [{"_id": "q1", "text": "query"}]
        ),
        "/datasets/BeIR/toy-qrels/resolve/main/test.tsv": b"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
    }

    def fake_fetch(url, timeout):
        return files[url.removeprefix(dl.HF)]

    def no_mirror(*args, **kwargs):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    monkeypatch.setattr(dl, "_from_zip", no_mirror)
    folder = dl.download("toy", tmp_path / "data")
    from ragbench import load_beir

    ds = load_beir(folder)
    assert ds.qrels == {"q1": {"d1": 1}}
    assert ds.corpus["d1"].text == "T\n\nbody"


def test_download_rejects_unknown_source(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        download("x", tmp_path, source="ftp")


def test_version_matches_package_metadata():
    from importlib.metadata import version

    import ragbench

    assert ragbench.__version__ == version("ragbench-eval")


def _run(tiny_path, out, *extra):
    argv = ["run", "--data", str(tiny_path), "--retriever", OVERLAP]
    return main([*argv, "--out", str(out), "--quiet", "--rounds", "50", *extra])


def test_recall_k_default_is_10_and_100(tiny_path, tmp_path):
    out = tmp_path / "r.json"
    assert _run(tiny_path, out) == 0
    data = json.loads(out.read_text())
    assert data["settings"]["recall_ks"] == [10, 100]
    assert sorted(data["metrics"]) == ["mrr@10", "ndcg@10", "recall@10", "recall@100"]


def test_recall_k_repeated_and_comma_separated(tiny_path, tmp_path):
    out = tmp_path / "r.json"
    assert _run(tiny_path, out, "--recall-k", "5,10", "--recall-k", "100", "--recall-k", "5") == 0
    data = json.loads(out.read_text())
    assert data["settings"]["recall_ks"] == [5, 10, 100]
    assert {"recall@5", "recall@10", "recall@100"} <= set(data["metrics"])
    assert set(data["per_query"]["recall@5"]) == set(data["per_query"]["recall@10"])
    # Recall can only grow with the cutoff.
    for q, r5 in data["per_query"]["recall@5"].items():
        assert r5 <= data["per_query"]["recall@10"][q] <= data["per_query"]["recall@100"][q]


def test_recall_k_only_five(tiny_path, tmp_path):
    out = tmp_path / "r.json"
    assert _run(tiny_path, out, "--recall-k", "5") == 0
    metrics = json.loads(out.read_text())["metrics"]
    assert "recall@5" in metrics and "recall@10" not in metrics


def test_recall_k_errors(tiny_path, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert _run(tiny_path, out, "--recall-k", "five") == 2
    assert "whole numbers" in capsys.readouterr().err
    assert _run(tiny_path, out, "--recall-k", "0") == 2
    assert "at least 1" in capsys.readouterr().err
    assert _run(tiny_path, out, "--recall-k", ",") == 2
    assert "at least one cutoff" in capsys.readouterr().err
    assert _run(tiny_path, out, "--recall-k", "200") == 2
    assert "shallower than the deepest cutoff" in capsys.readouterr().err
    assert not out.exists()
