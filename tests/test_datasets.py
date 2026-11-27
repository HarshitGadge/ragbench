import pytest

from ragbench.datasets import load_beir, read_qrels


def test_load_tiny(tiny):
    assert tiny.name == "tiny"
    assert len(tiny.corpus) == 6
    # q4 has no judgements, so it is dropped
    assert set(tiny.queries) == {"q1", "q2", "q3"}
    assert tiny.qrels["q1"] == {"d1": 2, "d2": 1}
    # a grade of 0 is "judged not relevant"
    assert "d2" not in tiny.qrels["q3"]
    assert tiny.summary() == {
        "name": "tiny",
        "split": "test",
        "documents": 6,
        "queries": 3,
        "judgements": 6,
    }


def test_document_text_joins_title(tiny):
    assert tiny.corpus["d1"].text.startswith("Coffee and sleep\n\n")
    assert tiny.corpus["d4"].text.startswith("Marathon")  # empty title


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_beir(tmp_path)


def test_qrels_without_header(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("q1\td1\t1\nq1\td2\t0\n")
    assert read_qrels(p) == {"q1": {"d1": 1}}


def test_qrels_bad_score(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("q1\td1\t1\nq1\td2\tmaybe\n")
    with pytest.raises(ValueError, match="not a number"):
        read_qrels(p)


def test_judged_query_missing_from_queries(tmp_path, tiny_path):
    import shutil

    folder = tmp_path / "broken"
    shutil.copytree(tiny_path, folder)
    with (folder / "qrels" / "test.tsv").open("a") as fh:
        fh.write("q99\td1\t1\n")
    with pytest.raises(ValueError, match="missing from queries"):
        load_beir(folder)
