"""Built-in retrievers. Use them as baselines or as examples of the interface.

ragbench run --data data/scifact --retriever ragbench.retrievers:BM25
pip install 'ragbench-eval[dense]'
ragbench run --data data/scifact --retriever ragbench.retrievers:Dense
"""

from ragbench.retrievers.bm25 import BM25
from ragbench.retrievers.dense import Dense

__all__ = ["BM25", "Dense"]
