"""``ragbench`` command line: download, info, run, compare, gate, report, faithfulness."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from ragbench import __version__, ragtruth
from ragbench.compare import compare_runs
from ragbench.datasets import load_beir
from ragbench.download import KNOWN, download
from ragbench.faithfulness import load_answers, load_judge, score_answers
from ragbench.gate import MODES, gate
from ragbench.report import DEFAULT_METRICS, markdown_table
from ragbench.retriever import load_retriever, parse_params
from ragbench.runner import RunResult, evaluate


def _cmd_download(args: argparse.Namespace) -> int:
    if args.name == ragtruth.NAME:
        out = ragtruth.build(args.dest)
        print((out.parent / "about.json").read_text(encoding="utf-8").strip())
        return 0
    folder = download(args.name, args.dest, url=args.url, source=args.source, split=args.split)
    print(json.dumps(load_beir(folder, args.split).summary()))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    if args.data is None:
        for name, about in KNOWN.items():
            print(f"{name:<12} {about}")
        print(f"{ragtruth.NAME:<12} {ragtruth.ABOUT}")
        return 0
    print(json.dumps(load_beir(args.data, args.split).summary(), indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = load_beir(args.data, args.split)
    params = parse_params(args.param)
    retriever = load_retriever(args.retriever, params)
    name = args.name or args.retriever.rpartition(":")[2]

    def progress(i: int, n: int) -> None:
        if not args.quiet and (i == n or i % 100 == 0):
            print(f"  {i}/{n} queries", file=sys.stderr, flush=True)

    result = evaluate(
        retriever,
        dataset,
        name=name,
        depth=args.depth,
        limit=args.limit,
        bootstrap_rounds=args.rounds,
        seed=args.seed,
        progress=progress,
        ignore_identical_ids=not args.keep_identical_ids,
        meta={
            "retriever": args.retriever,
            "params": params,
            "command": "ragbench " + shlex.join(args.argv),
        },
    )
    out = Path(args.out) if args.out else Path("results") / f"{dataset.name}__{name}.json"
    result.save(out)
    print(markdown_table([result]), end="")
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    base, cand = RunResult.load(args.baseline), RunResult.load(args.candidate)
    metrics = args.metric or ["ndcg@10"]
    rows = [compare_runs(base, cand, m, rounds=args.rounds, seed=args.seed) for m in metrics]
    if args.json:
        print(json.dumps([r.as_dict() for r in rows], indent=2))
        return 0
    for r in rows:
        flag = "significant" if r.significant else "not significant"
        print(
            f"{r.metric}: {r.baseline} {r.baseline_mean:.4f} -> "
            f"{r.candidate} {r.candidate_mean:.4f}, "
            f"diff {r.diff.mean:+.4f} (95% CI {r.diff.low:+.4f} to {r.diff.high:+.4f}, "
            f"{r.queries} queries, {flag})"
        )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    base, cand = RunResult.load(args.baseline), RunResult.load(args.candidate)
    result = gate(
        base,
        cand,
        metrics=args.metric or ["ndcg@10"],
        max_drop=args.max_drop,
        mode=args.mode,
        rounds=args.rounds,
        seed=args.seed,
    )
    for check in result.checks:
        print(check.line())
    summary = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(result.markdown())
    return 0 if result.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    runs = [RunResult.load(p) for p in args.results]
    runs.sort(key=lambda r: (r.dataset.get("name", ""), r.name))
    print(markdown_table(runs, args.metric or DEFAULT_METRICS), end="")
    return 0


def _cmd_faithfulness(args: argparse.Namespace) -> int:
    records = load_answers(args.answers)
    params = parse_params(args.param)
    judge = load_judge(args.judge, params)

    def progress(i: int, n: int) -> None:
        if not args.quiet and (i == n or i % 100 == 0):
            print(f"  {i}/{n} answers", file=sys.stderr, flush=True)

    result = score_answers(
        judge,
        records,
        judge_spec=args.judge,
        params=params,
        name=args.name or "",
        threshold=args.threshold,
        rounds=args.rounds,
        seed=args.seed,
        progress=progress,
        meta={"answers_file": str(args.answers), "command": "ragbench " + shlex.join(args.argv)},
    )
    s = result.summary
    f = s["faithfulness"]
    print(
        f"faithfulness {f['mean']:.4f} (95% CI {f['ci'][0]:.4f} to {f['ci'][1]:.4f}), "
        f"{s['scored']} answers, {s['claims']} claims, judge {result.name}, "
        f"threshold {result.threshold}"
    )
    if s["no_claims"]:
        print(f"{s['no_claims']} answers had no claim to score and were left out")
    if result.agreement:
        a = result.agreement
        print(
            f"agreement with {a['labelled']} human labels: ROC AUC {a['auc_faithfulness']:.4f} "
            f"(mean support {a['auc_mean_support']:.4f}), "
            f"accuracy {a['accuracy_all_supported']:.4f} calling an answer faithful when every "
            "claim is supported"
        )
    if args.out:
        result.save(args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragbench", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"ragbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download a BEIR dataset, or ragtruth-qa")
    p.add_argument(
        "name", help="BEIR dataset name, e.g. scifact, or ragtruth-qa (see `ragbench info`)"
    )
    p.add_argument("--dest", default="data")
    p.add_argument("--split", default="test")
    p.add_argument("--url", default=None, help="override the archive URL")
    p.add_argument("--source", choices=["auto", "ukp", "hf"], default="auto")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("info", help="list known datasets, or summarise one folder")
    p.add_argument("data", nargs="?", type=Path)
    p.add_argument("--split", default="test")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("run", help="evaluate a retriever on a dataset")
    p.add_argument("--data", required=True, type=Path, help="BEIR-format dataset folder")
    p.add_argument("--retriever", required=True, help="module:Name or path/file.py:Name")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--name", default=None, help="label for this run (default: Name)")
    p.add_argument("--split", default="test")
    p.add_argument("--depth", type=int, default=100, help="results requested per query")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--rounds", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="result JSON path")
    p.add_argument(
        "--keep-identical-ids",
        action="store_true",
        help="score a retrieved document whose id equals the query id (BEIR drops these)",
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("compare", help="paired bootstrap between two result files")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--metric", action="append", default=None)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser(
        "gate",
        help="exit 1 when a candidate run drops below a baseline (for CI)",
        description="Compare a candidate run with a baseline run on the same queries and "
        "exit 1 if any metric dropped by more than --max-drop.",
    )
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--metric", action="append", default=None, help="repeatable; ndcg@10")
    p.add_argument("--max-drop", type=float, default=0.01, help="allowed drop in the metric (0.01)")
    p.add_argument(
        "--mode",
        choices=list(MODES),
        default="mean",
        help="mean: fail on the mean change; ci: fail only if the whole 95%% CI is below",
    )
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--summary",
        default=None,
        help="append a Markdown summary here (default: $GITHUB_STEP_SUMMARY if set)",
    )
    p.set_defaults(func=_cmd_gate)

    p = sub.add_parser(
        "faithfulness",
        help="score whether generated answers are supported by their retrieved contexts",
        description="Split each answer into claims, ask a judge whether the contexts support "
        "each claim, and report the share of supported claims with a bootstrap CI. With "
        "human labels in the answers file, also report how well the judge agrees with them.",
    )
    p.add_argument(
        "--answers", required=True, type=Path, help="JSONL: id, question, contexts, answer"
    )
    p.add_argument(
        "--judge",
        default="ragbench.judges:TokenOverlap",
        help="module:Name or path/file.py:Name (default: the lexical TokenOverlap baseline)",
    )
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--name", default=None, help="label for this run (default: judge Name)")
    p.add_argument("--threshold", type=float, default=0.5, help="support needed to count a claim")
    p.add_argument("--rounds", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write per-answer results to this JSON file")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_faithfulness)

    p = sub.add_parser("report", help="Markdown table of result files")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--metric", action="append", default=None)
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = argv
    try:
        return args.func(args)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        AttributeError,
    ) as exc:
        print(f"ragbench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
