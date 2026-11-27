"""Built-in retrievers. Use them as baselines or as examples of the interface.

ragbench run --data data/scifact --retriever ragbench.retrievers:BM25
pip install 'ragbench[dense]'
ragbench run --data data/scifact --retriever ragbench.retrievers:Dense
ragbench run --data data/scifact --retriever ragbench.retrievers:Hybrid
ragbench run --data data/scifact --retriever ragbench.retrievers:Rerank
"""

from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import Dense
from ragbench.retrievers.hybrid import Hybrid
from ragbench.retrievers.rerank import Rerank

__all__ = ["BM25", "Dense", "Hybrid", "Rerank"]
