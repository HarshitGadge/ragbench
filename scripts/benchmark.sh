#!/usr/bin/env bash
# Reproduce every number in the README's Results section.
#
#   pip install -e '.[dense,stem,hf]'
#   bash scripts/benchmark.sh            # or DATASETS="scifact" bash scripts/benchmark.sh
#   python scripts/update_readme.py      # regenerate the README table from results/
#
# The published results/ were produced by the "Benchmark" GitHub Actions workflow,
# which runs exactly this script; each result file records the run's URL and commit.
set -euo pipefail

DATASETS="${DATASETS:-scifact nfcorpus arguana}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-small-en-v1.5}"

for d in $DATASETS; do
  echo "== $d"
  ragbench download "$d" --dest data
  ragbench run --data "data/$d" --retriever ragbench.retrievers:BM25 \
    --name bm25 --out "results/$d/bm25.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:BM25 --param stem=true \
    --name bm25-stem --out "results/$d/bm25-stem.json" --quiet
  ragbench run --data "data/$d" --retriever ragbench.retrievers:Dense \
    --param "model=$DENSE_MODEL" --name dense-bge-small --out "results/$d/dense-bge-small.json" --quiet
  ragbench compare "results/$d/bm25-stem.json" "results/$d/dense-bge-small.json" \
    --metric ndcg@10 --metric recall@100 --json > "results/$d/compare.json"
done
