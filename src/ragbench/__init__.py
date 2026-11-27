"""RAGBench: a small, dependency-free harness for evaluating retrievers on BEIR-format data."""

from ragbench.datasets import Dataset, Document, load_beir
from ragbench.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from ragbench.retriever import Hit, Retriever
from ragbench.runner import RunResult, evaluate
from ragbench.stats import bootstrap_ci, paired_bootstrap

__version__ = "1.1.0"

__all__ = [
    "Dataset",
    "Document",
    "Hit",
    "Retriever",
    "RunResult",
    "__version__",
    "bootstrap_ci",
    "evaluate",
    "load_beir",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap",
    "recall_at_k",
]
