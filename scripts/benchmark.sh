#!/usr/bin/env bash
# Reproduce every number in the README's Results section.
#
#   pip install -e '.[dense,stem,hf]'
#   bash scripts/benchmark.sh            # or DATASETS="scifact" bash scripts/benchmark.sh
#   python scripts/update_readme.py      # regenerate the README table from results/
#
# The published results/ were produced by the "Benchmark" GitHub Actions workflow,
# which runs exactly this script; each result file records the run's URL and commit.
#
# Dense, Hybrid and Rerank share one corpus embedding per dataset through
# EMBED_CACHE, so the corpus is embedded once rather than three times.
set -euo pipefail

DATASETS="${DATASETS:-scifact nfcorpus arguana}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-small-en-v1.5}"
RERANK_MODEL="${RERANK_MODEL:-Xenova/ms-marco-MiniLM-L-6-v2}"
RERANK_CANDIDATES="${RERANK_CANDIDATES:-30}"
EMBED_CACHE="${EMBED_CACHE:-cache/embeddings}"

for d in $DATASETS; do
  echo "== $d"
  out="results/$d"
  ragbench download "$d" --dest data
  ragbench run --data "data/$d" --retriever ragbench.retrievers:BM25 \
    --name bm25 --out "$out/bm25.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:BM25 --param stem=true \
    --name bm25-stem --out "$out/bm25-stem.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:Dense \
    --param "model=$DENSE_MODEL" --param "cache=$EMBED_CACHE" \
    --name dense-bge-small --out "$out/dense-bge-small.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:Hybrid \
    --param "model=$DENSE_MODEL" --param "cache=$EMBED_CACHE" \
    --name hybrid --out "$out/hybrid.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:Rerank \
    --param first_stage=hybrid --param "model=$RERANK_MODEL" \
    --param "candidates=$RERANK_CANDIDATES" \
    --param "dense_model=$DENSE_MODEL" --param "cache=$EMBED_CACHE" \
    --name rerank-minilm --out "$out/rerank-minilm.json" --quiet

  ragbench compare "$out/bm25-stem.json" "$out/dense-bge-small.json" \
    --metric ndcg@10 --metric recall@100 --json > "$out/compare-bm25-stem__dense-bge-small.json"
  ragbench compare "$out/dense-bge-small.json" "$out/hybrid.json" \
    --metric ndcg@10 --metric recall@100 --json > "$out/compare-dense-bge-small__hybrid.json"
  # Reranking reorders only the top RERANK_CANDIDATES, so Recall@100 can't move.
  ragbench compare "$out/hybrid.json" "$out/rerank-minilm.json" \
    --metric ndcg@10 --json > "$out/compare-hybrid__rerank-minilm.json"
done
