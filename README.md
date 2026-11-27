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
- **A CI gate.** A GitHub Action fails a pull request when retrieval quality drops
  below a committed baseline, the way a coverage check does for tests.
- **Answer faithfulness.** Scores whether generated answers are supported by their
  retrieved contexts, with a pluggable judge and a check against human labels.

Every number in this README comes from a result file in [`results/`](results), produced
by [`scripts/benchmark.sh`](scripts/benchmark.sh) in the Benchmark workflow. The table
below is generated from those files, and CI fails if it drifts.

## Results

Built-in retrievers on three BEIR test sets: SciFact, NFCorpus
and ArguAna. Dense uses `BAAI/bge-small-en-v1.5` through fastembed with no query
instruction; BM25 uses k1=0.9, b=0.4. Latency is per query on a GitHub-hosted runner,
and Dense batches its queries, so its p50 is the batch time divided by the query count.
Hybrid fuses stemmed BM25 and Dense with reciprocal rank fusion (`rrf_k=60`, top 100 of
each), and rerank-minilm re-scores Hybrid's top 30 with `Xenova/ms-marco-MiniLM-L-6-v2`;
their latency includes every stage.

Dense beats stemmed BM25 on nDCG@10 on all three datasets, significantly on SciFact and
ArguAna. Hybrid helps only on NFCorpus and is worse than Dense on ArguAna, where BM25 is far
weaker. The cross-encoder lowers nDCG@10 on SciFact and ArguAna and costs 1.4 to 3.3 s per
query on a CPU runner; ArguAna's queries are whole arguments, unlike the short questions
the reranker was trained on.

<!-- results:start -->
| dataset | retriever | ndcg@10 | recall@10 | recall@100 | mrr@10 | queries | p50 ms |
|---|---|---|---|---|---|---|---|
| arguana | bm25 | 0.4052 (0.387 to 0.424) | 0.6814 (0.657 to 0.707) | 0.9317 (0.918 to 0.945) | 0.3203 (0.302 to 0.340) | 1406 | 21.79 |
| arguana | bm25-stem | 0.4012 (0.382 to 0.420) | 0.6643 (0.639 to 0.689) | 0.9324 (0.919 to 0.946) | 0.3198 (0.301 to 0.339) | 1406 | 31.07 |
| arguana | dense-bge-small | 0.5970 (0.578 to 0.616) | 0.8535 (0.835 to 0.871) | 0.9844 (0.978 to 0.991) | 0.5163 (0.495 to 0.537) | 1406 | 208.21 |
| arguana | hybrid | 0.5506 (0.532 to 0.570) | 0.8329 (0.812 to 0.851) | 0.9851 (0.979 to 0.992) | 0.4627 (0.444 to 0.484) | 1406 | 236.79 |
| arguana | rerank-minilm | 0.4394 (0.421 to 0.459) | 0.7340 (0.712 to 0.757) | 0.9851 (0.979 to 0.992) | 0.3491 (0.331 to 0.369) | 1406 | 3317.82 |
| nfcorpus | bm25 | 0.3061 (0.272 to 0.340) | 0.1466 (0.122 to 0.173) | 0.2360 (0.208 to 0.266) | 0.5106 (0.462 to 0.561) | 323 | 0.14 |
| nfcorpus | bm25-stem | 0.3208 (0.287 to 0.355) | 0.1523 (0.127 to 0.179) | 0.2461 (0.218 to 0.276) | 0.5231 (0.474 to 0.572) | 323 | 0.22 |
| nfcorpus | dense-bge-small | 0.3385 (0.306 to 0.373) | 0.1609 (0.137 to 0.187) | 0.3057 (0.277 to 0.336) | 0.5305 (0.484 to 0.577) | 323 | 2.54 |
| nfcorpus | hybrid | 0.3640 (0.329 to 0.400) | 0.1730 (0.148 to 0.201) | 0.3132 (0.285 to 0.343) | 0.5730 (0.523 to 0.621) | 323 | 4.83 |
| nfcorpus | rerank-minilm | 0.3594 (0.325 to 0.394) | 0.1684 (0.144 to 0.196) | 0.3132 (0.285 to 0.343) | 0.5865 (0.538 to 0.634) | 323 | 1429.31 |
| scifact | bm25 | 0.6635 (0.620 to 0.710) | 0.7834 (0.737 to 0.829) | 0.8859 (0.851 to 0.922) | 0.6316 (0.585 to 0.681) | 300 | 1.40 |
| scifact | bm25-stem | 0.6804 (0.635 to 0.724) | 0.8030 (0.758 to 0.846) | 0.9220 (0.891 to 0.951) | 0.6475 (0.600 to 0.694) | 300 | 2.02 |
| scifact | dense-bge-small | 0.7201 (0.677 to 0.760) | 0.8452 (0.806 to 0.883) | 0.9533 (0.927 to 0.977) | 0.6846 (0.637 to 0.728) | 300 | 12.79 |
| scifact | hybrid | 0.7362 (0.694 to 0.777) | 0.8562 (0.815 to 0.893) | 0.9650 (0.943 to 0.983) | 0.7047 (0.659 to 0.750) | 300 | 15.21 |
| scifact | rerank-minilm | 0.6990 (0.656 to 0.741) | 0.8356 (0.793 to 0.876) | 0.9650 (0.943 to 0.983) | 0.6656 (0.619 to 0.712) | 300 | 2451.57 |


Means over test queries with 95% bootstrap intervals; ragbench 1.1.1, commit f4cb14b, workflow run [45949717691](https://github.com/HarshitGadge/ragbench/actions/runs/45949717691).


Paired bootstrap, dense-bge-small minus bm25-stem, same queries:

- arguana, ndcg@10: +0.1958 (95% CI +0.1774 to +0.2146, 1406 queries, interval excludes 0)
- arguana, recall@100: +0.0519 (95% CI +0.0391 to +0.0647, 1406 queries, interval excludes 0)
- nfcorpus, ndcg@10: +0.0177 (95% CI -0.0040 to +0.0390, 323 queries, interval includes 0)
- nfcorpus, recall@100: +0.0596 (95% CI +0.0405 to +0.0795, 323 queries, interval excludes 0)
- scifact, ndcg@10: +0.0397 (95% CI +0.0051 to +0.0730, 300 queries, interval excludes 0)
- scifact, recall@100: +0.0313 (95% CI -0.0013 to +0.0660, 300 queries, interval includes 0)

Paired bootstrap, hybrid minus dense-bge-small, same queries:

- arguana, ndcg@10: -0.0464 (95% CI -0.0601 to -0.0320, 1406 queries, interval excludes 0)
- arguana, recall@100: +0.0007 (95% CI -0.0043 to +0.0057, 1406 queries, interval includes 0)
- nfcorpus, ndcg@10: +0.0254 (95% CI +0.0107 to +0.0400, 323 queries, interval excludes 0)
- nfcorpus, recall@100: +0.0076 (95% CI -0.0039 to +0.0178, 323 queries, interval includes 0)
- scifact, ndcg@10: +0.0161 (95% CI -0.0079 to +0.0401, 300 queries, interval includes 0)
- scifact, recall@100: +0.0117 (95% CI -0.0100 to +0.0333, 300 queries, interval includes 0)

Paired bootstrap, rerank-minilm minus hybrid, same queries:

- arguana, ndcg@10: -0.1112 (95% CI -0.1279 to -0.0937, 1406 queries, interval excludes 0)
- nfcorpus, ndcg@10: -0.0045 (95% CI -0.0183 to +0.0080, 323 queries, interval includes 0)
- scifact, ndcg@10: -0.0372 (95% CI -0.0652 to -0.0085, 300 queries, interval excludes 0)
<!-- results:end -->

Reproduce:

```bash
pip install -e '.[dense,stem,hf]'
bash scripts/benchmark.sh
python scripts/update_readme.py
python scripts/leaderboard.py      # leaderboard.json, the same results for web pages
```

## Install

```bash
pip install ragbench-eval

# with the built-in dense retriever, stemming and the Hugging Face download fallback
pip install 'ragbench-eval[dense,stem,hf]'
```

[ragbench-eval on PyPI](https://pypi.org/project/ragbench-eval/). To install straight from
a tag instead: `pip install "git+https://github.com/HarshitGadge/ragbench@v1.1.1"`.

The distribution is named `ragbench-eval`, because PyPI reserves names too close to the
existing `rag-bench` package; you still `import ragbench` and run `ragbench`.
The core package has no runtime dependencies. Extras: `dense` (fastembed, numpy),
`stem` (snowballstemmer), `hf` (pyarrow).

## Quick start

```bash
ragbench info                       # datasets worth starting with
ragbench download scifact           # into ./data/scifact
ragbench run --data data/scifact --retriever examples/overlap.py:WordOverlap
```

`run` prints a Markdown table and writes `results/<dataset>__<name>.json`.
It reports nDCG@10, MRR@10, Recall@10 and Recall@100. To report recall at other
cutoffs, for example the top 5 that an agent actually reads, pass `--recall-k`: it can
be repeated or comma-separated, and replaces the default 10 and 100.

```bash
ragbench run --data data/scifact --retriever ragbench.retrievers:BM25 --recall-k 5,10,100
```

## Built-in retrievers

Four retrievers ship with RAGBench, and they double as examples of the interface.

| retriever | spec | needs |
|---|---|---|
| BM25 (Lucene scoring, k1=0.9, b=0.4) | `ragbench.retrievers:BM25` | nothing |
| Dense, cosine over normalised embeddings | `ragbench.retrievers:Dense` | `pip install 'ragbench-eval[dense]'` |
| Hybrid, reciprocal rank fusion of BM25 and Dense | `ragbench.retrievers:Hybrid` | `pip install 'ragbench-eval[dense]'` |
| Rerank, a cross-encoder over a first stage's top hits | `ragbench.retrievers:Rerank` | `pip install 'ragbench-eval[dense]'` |

```bash
ragbench run --data data/scifact --retriever ragbench.retrievers:BM25 --param stem=true
ragbench run --data data/scifact --retriever ragbench.retrievers:Dense \
  --param model=BAAI/bge-small-en-v1.5
```

BM25 takes `k1`, `b`, `stopwords` (Lucene's English list, on by default) and `stem`
(Snowball, `pip install 'ragbench-eval[stem]'`). Its tokeniser is simpler than an
Elasticsearch analyser, so its scores will not match Elasticsearch-based BEIR
baselines exactly. Dense takes `model` (any [fastembed](https://github.com/qdrant/fastembed)
text model), `query_prefix` for models that expect one, and `batch_size`; from Python
you can pass `encoder=` with any object that has `embed_documents` and `embed_query`.

Hybrid fuses BM25 and Dense with reciprocal rank fusion (score = Σ 1/(`rrf_k` + rank),
`rrf_k=60` as in Cormack, Clarke and Buettcher, SIGIR 2009) over each one's top
`candidates` (100). Rerank re-scores the top `candidates` (30) of a `first_stage`
(`bm25`, `dense` or `hybrid`) with a fastembed cross-encoder (`model`, default
`Xenova/ms-marco-MiniLM-L-6-v2`) and keeps the rest in first-stage order, so it can
change nDCG@10 but not Recall@100. Dense, Hybrid and Rerank take `cache=<dir>` to save
the corpus embedding as a `.npy` file keyed by the model and the exact corpus text, so
several runs on one dataset embed it once.

```bash
ragbench run --data data/scifact --retriever ragbench.retrievers:Hybrid --param cache=cache/embeddings
ragbench run --data data/scifact --retriever ragbench.retrievers:Rerank \
  --param first_stage=hybrid --param candidates=30 --param cache=cache/embeddings
```

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

## Gate pull requests on retrieval quality

`ragbench gate` compares a candidate run with a baseline run on the same queries and
exits 1 when a metric dropped by more than `--max-drop`:

```bash
ragbench gate baseline.json candidate.json --metric ndcg@10 --max-drop 0.01
```

With `--mode mean` (the default) it fails on the mean change. With `--mode ci` it fails
only when the whole 95% interval for the change is below `-max-drop`, so a drop that
could be noise doesn't fail the build. Runs that differ in split, depth, nDCG cutoff
or identical-id handling aren't comparable, and the gate refuses them.

The same check runs as a GitHub Action. Commit a small BEIR-format evaluation set and a
baseline run, then add a workflow:

```bash
ragbench run --data eval/my-set --retriever my_app.search:Retriever --out eval/baseline.json
```

```yaml
# .github/workflows/retrieval.yml
on: pull_request
jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: HarshitGadge/ragbench@v1.1.0
        with:
          install: pip install -e .   # so the action can import my_app
          data: eval/my-set
          retriever: my_app.search:Retriever
          baseline: eval/baseline.json
          metrics: ndcg@10,recall@100
          max-drop: "0.01"
```

The job summary shows a table of baseline, candidate and change with its interval.
A retriever given as `path/to/file.py:Name` needs no `install`. Inputs `params` (one
`KEY=VALUE` per line), `mode`, `split`, `depth`, `recall-k` (cutoffs such as `5,10,100`,
to gate on `recall@5`), `extras` and `python-version` are
optional too; see [`action.yml`](action.yml). Re-run the baseline
command and commit the file when you mean to change the baseline.

## Score answer faithfulness

Retrieval metrics say whether the right passages were found, not whether the answer
stuck to them. `ragbench faithfulness` scores generated answers: each answer is split
into sentence claims, a judge rates how well the retrieved contexts support each claim,
and the answer scores the share of claims at or above `--threshold` (0.5). The run
reports the mean over answers with a bootstrap 95% interval.

```bash
ragbench faithfulness --answers answers.jsonl --out results/faithfulness.json
```

`answers.jsonl` has one answer per line:

```json
{"id": "q1", "question": "...", "contexts": ["retrieved passage", "..."], "answer": "..."}
```

Add `"label": 1` (faithful) or `0` (not) from a human reviewer, and the run also reports
how well the judge agrees with the labels: ROC AUC, and accuracy when an answer counts
as faithful only if every claim is supported. Check a judge this way before trusting
its scores.

The default judge, `ragbench.judges:TokenOverlap`, needs nothing installed. It is a
lexical baseline: a claim counts as supported when its content words appear in one
context. It cannot see paraphrase ("the CFPB" for "the Bureau") and only partly sees
negation, so it is the floor a real judge has to beat. A judge is any object with
`support(claim, contexts, question)` returning a score in [0, 1], loaded with
`--judge module:Name` or `--judge path/to/file.py:Name`, like a retriever; a
`support_batch(items)` method is used when present.

`ragbench.judges:NLI` is a model judge (`pip install 'ragbench-eval[nli]'`). It runs a
natural-language-inference model locally, by default DeBERTa-v3-base trained on MNLI,
FEVER and ANLI, and takes a claim's support as the highest probability that one
context entails it. Choose another three-way NLI model with `--param model=<hub id>`.

To check a judge against people, `ragbench download ragtruth-qa` builds 875 answers
from the question-answering test split of [RAGTruth](https://github.com/ParticleMedia/RAGTruth)
(MIT licence), written by six LLMs and labelled unfaithful wherever annotators marked
an unsupported span. The files come from a pinned commit and are checked against their
SHA-256, so every run scores the same answers:

```bash
ragbench download ragtruth-qa --dest data
ragbench faithfulness --answers data/ragtruth-qa/answers.jsonl --judge ragbench.judges:NLI
```

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
