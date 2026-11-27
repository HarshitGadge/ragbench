"""The retriever interface, and how a retriever is found from the command line.

A retriever is any object with a ``search(query, k)`` method. It may also have:

* ``index(corpus)``: called once with ``{doc_id: Document}`` before any search.
* ``search_batch(queries, k)``: used instead of ``search`` when present.

``search`` may return document ids, :class:`Hit` objects, or ``(doc_id, score)``
pairs, best first. Repeated ids (several chunks of one document) are fine: a
document counts once, at its best rank.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragbench.datasets import Document


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float | None = None


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int) -> Sequence[Hit | str | tuple[str, float]]: ...


def doc_ids(results: Sequence[Any]) -> list[str]:
    """Normalise whatever ``search`` returned into a list of document ids."""
    ids: list[str] = []
    for item in results:
        if isinstance(item, Hit):
            ids.append(str(item.doc_id))
        elif isinstance(item, str):
            ids.append(item)
        elif isinstance(item, tuple | list) and item:
            ids.append(str(item[0]))
        elif hasattr(item, "doc_id"):
            ids.append(str(item.doc_id))
        else:
            raise TypeError(
                f"search() returned {type(item).__name__}; expected a doc id, Hit or (id, score)"
            )
    return ids


def index_if_supported(retriever: Any, corpus: Mapping[str, Document]) -> None:
    index = getattr(retriever, "index", None)
    if callable(index):
        index(corpus)


def parse_params(pairs: Sequence[str]) -> dict[str, Any]:
    """Turn ``["k1=0.9", "name=bm25"]`` into ``{"k1": 0.9, "name": "bm25"}``.

    Values are read as JSON when they parse (numbers, booleans, lists), else as text.
    """
    params: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--param expects key=value, got {pair!r}")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def _import_target(spec: str) -> Any:
    module_part, sep, attr = spec.rpartition(":")
    if not sep or not module_part or not attr:
        raise ValueError(
            f"retriever spec must look like 'module:Name' or 'file.py:Name', got {spec!r}"
        )
    if module_part.endswith(".py"):
        path = Path(module_part).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        mod_name = f"ragbench_user_{path.stem}"
        loader_spec = importlib.util.spec_from_file_location(mod_name, path)
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[mod_name] = module
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    target = module
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def load_retriever(spec: str, params: Mapping[str, Any] | None = None) -> Any:
    """Build a retriever from ``module:Name`` or ``path/to/file.py:Name``.

    ``Name`` may be a class, a factory function, or an existing instance. Classes and
    functions are called with ``params`` as keyword arguments.
    """
    target = _import_target(spec)
    params = dict(params or {})
    if inspect.isclass(target) or inspect.isfunction(target):
        retriever = target(**params)
    elif params:
        raise ValueError(f"{spec} is an instance, so it cannot take --param values")
    else:
        retriever = target
    if not callable(getattr(retriever, "search", None)):
        raise TypeError(
            f"{spec} built a {type(retriever).__name__} with no search(query, k) method"
        )
    return retriever
