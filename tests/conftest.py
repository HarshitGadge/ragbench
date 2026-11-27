from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "tests" / "fixtures" / "tiny"


@pytest.fixture
def tiny_path() -> Path:
    return TINY


@pytest.fixture
def tiny():
    from ragbench import load_beir

    return load_beir(TINY)
