# RAGBench

Evaluate any retriever on [BEIR](https://github.com/beir-cellar/beir)-format data and
get numbers you can defend: nDCG@10, Recall@10/100 and MRR@10 per query, a bootstrap
95% confidence interval on every mean, and a paired bootstrap when you compare two runs.

- **Bring your own retriever.** Anything with a `search(query, k)` method plugs in,
  from a module or a single `.py` file. No base class, no framework.
- **No runtime dependencies.** The core is the Python standard library.
- **Checked against `trec_eval`.** The metric tests compare every score with
  `pytrec_eval` on randomised runs with graded judgements.
- **Reproducible results.** Each result file records the exact command, parameters,
  seed, Python and RAGBench versions, and every per-query score.

Every number in this README comes from a result file in [`results/`](results), produced
by [`scripts/benchmark.sh`](scripts/benchmark.sh) in the Benchmark workflow. The table
below is generated from those files, and CI fails if it drifts.

## Results

Built-in retrievers on three BEIR test sets: SciFact, NFCorpus
and ArguAna. Dense uses `BAAI/bge-small-en-v1.5` through fastembed with no query
instruction; BM25 uses k1=0.9, b=0.4. Latency is per query on a GitHub-hosted runner,
and Dense batches its queries, so its p50 is the batch time divided by the query count.

<!-- results:start -->
No results yet. Run `bash scripts/benchmark.sh`.
<!-- results:end -->

Reproduce:

```bash
pip install -e '.[dense,stem,hf]'
bash scripts/benchmark.sh
python scripts/update_readme.py
```

## Install

```bash
pip install git+https://github.com/HarshitGadge/ragbench
```

## Quick start

```bash
ragbench info                       # datasets worth starting with
ragbench download scifact           # into ./data/scifact
ragbench run --data data/scifact --retriever examples/overlap.py:WordOverlap
```

`run` prints a Markdown table and writes `results/<dataset>__<name>.json`.

## Built-in retrievers

Two baselines ship with RAGBench, and they double as examples of the interface.

| retriever | spec | needs |
|---|---|---|
| BM25 (Lucene scoring, k1=0.9, b=0.4) | `ragbench.retrievers:BM25` | nothing |
| Dense, cosine over normalised embeddings | `ragbench.retrievers:Dense` | `pip install 'ragbench[dense]'` |

```bash
ragbench run --data data/scifact --retriever ragbench.retrievers:BM25 --param stem=true
ragbench run --data data/scifact --retriever ragbench.retrievers:Dense \
  --param model=BAAI/bge-small-en-v1.5
```

BM25 takes `k1`, `b`, `stopwords` (Lucene's English list, on by default) and `stem`
(Snowball, `pip install 'ragbench[stem]'`). Its tokeniser is simpler than an
Elasticsearch analyser, so its scores will not match Elasticsearch-based BEIR
baselines exactly. Dense takes `model` (any [fastembed](https://github.com/qdrant/fastembed)
text model), `query_prefix` for models that expect one, and `batch_size`; from Python
you can pass `encoder=` with any object that has `embed_documents` and `embed_query`.

## Plug in a retriever

```python
# my_retriever.py
class MyRetriever:
    def __init__(self, top_n: int = 100):
        self.top_n = top_n

    def index(self, corpus):  # optional: {doc_id: Document(id, title, body)}
        ...

    def search(self, query: str, k: int):
        return [("doc-17", 12.3), ("doc-4", 11.0)]  # ids, Hit objects or (id, score)
```

```bash
ragbench run --data data/scifact --retriever my_retriever.py:MyRetriever --param top_n=50
```

`--param key=value` values are parsed as JSON when they can be (`0.5`, `true`, `[1,2]`).
A retriever can also define `search_batch(queries, k)`, which RAGBench uses instead of
`search` when present. Returning several chunks of the same document is fine: a
document counts once, at the rank of its best chunk.

From Python:

```python
from ragbench import load_beir, evaluate

dataset = load_beir("data/scifact")
result = evaluate(MyRetriever(), dataset, name="mine")
print(result.summary()["ndcg@10"])  # {'mean': ..., 'ci': [low, high], 'level': 0.95}
result.save("results/scifact__mine.json")
```

## Compare two runs

```bash
ragbench compare results/scifact__baseline.json results/scifact__mine.json --metric ndcg@10
```

The difference is resampled over queries (2,000 rounds, seeded), so the interval says
whether the change would survive a different sample of questions. Both runs must cover
the same queries.

## Make a results table

```bash
ragbench report results/*.json
```

## How the metrics are computed

| metric | definition |
|---|---|
| nDCG@10 | DCG with linear gains (the judgement grade) and log2 discounts, divided by the ideal DCG; the `ndcg_cut.10` that `trec_eval` and BEIR report |
| Recall@k | relevant documents in the top k, over all relevant documents |
| MRR@10 | 1 / rank of the first relevant document in the top 10, else 0 |

Judgements of 0 count as not relevant, and queries without a relevant document are
skipped, as in BEIR.

## Development

```bash
pip install -e '.[dev,stem]' numpy pytrec-eval-terrier
ruff check . && ruff format --check . && pytest
```

## Origin

The loader, metrics and bootstrap started as the retrieval evaluation in
[agentic-rag-service](https://github.com/HarshitGadge/agentic-rag-service) and were
generalised here so they are not tied to one service.

## License

MIT
