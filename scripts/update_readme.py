"""Rewrite the README's Results section from the files in results/.

    python scripts/update_readme.py           # rewrite README.md
    python scripts/update_readme.py --check   # exit 1 if README.md is out of date (CI)

Nothing between the markers is typed by hand: the table comes from the result files
and the comparison lines from each dataset's compare*.json files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragbench.report import markdown_table
from ragbench.runner import RunResult

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
# Optional second block: faithfulness judges checked against human labels.
F_START, F_END = "<!-- faithfulness:start -->", "<!-- faithfulness:end -->"


def _source(metas: list[dict]) -> str:
    ci_runs = sorted({m.get("ci_run") for m in metas if m.get("ci_run")})
    shas = sorted({m.get("git_sha")[:7] for m in metas if m.get("git_sha")})
    source = f"ragbench {', '.join(sorted({m.get('ragbench', '?') for m in metas}))}"
    if shas:
        source += f", commit {', '.join(shas)}"
    if ci_runs:
        source += ", workflow run " + ", ".join(f"[{u.rsplit('/', 1)[-1]}]({u})" for u in ci_runs)
    return source


def render(results_dir: Path) -> str:
    runs = sorted(
        (
            RunResult.load(p)
            for p in results_dir.glob("*/*.json")
            if not p.name.startswith("compare")
        ),
        key=lambda r: (r.dataset.get("name", ""), r.name),
    )
    if not runs:
        return "No results yet. Run `bash scripts/benchmark.sh`.\n"
    lines = [markdown_table(runs)]
    source = _source([r.meta for r in runs])
    lines.append(f"\nMeans over test queries with 95% bootstrap intervals; {source}.\n")

    # One block per (baseline, candidate) pair, listing every dataset that compared them.
    pairs: dict[tuple[str, str], list[str]] = {}
    for path in sorted(results_dir.glob("*/compare*.json"), key=lambda p: (p.name, p.parent.name)):
        for row in json.loads(path.read_text()):
            d = row["diff"]
            verdict = "interval excludes 0" if row["significant"] else "interval includes 0"
            pairs.setdefault((row["baseline"], row["candidate"]), []).append(
                f"- {path.parent.name}, {row['metric']}: {d['mean']:+.4f} "
                f"(95% CI {d['ci'][0]:+.4f} to {d['ci'][1]:+.4f}, "
                f"{row['queries']} queries, {verdict})"
            )
    for (baseline, candidate), rows in pairs.items():
        lines.append(f"\nPaired bootstrap, {candidate} minus {baseline}, same queries:\n")
        lines.extend(rows)
    if pairs:
        lines.append("")
    return "\n".join(lines)


def render_faithfulness(results_dir: Path) -> str:
    """Table of faithfulness runs in results_dir/faithfulness/<dataset>/<judge>.json."""
    runs = []
    for path in sorted((results_dir / "faithfulness").glob("*/*.json")):
        if path.name == "about.json":
            continue
        runs.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    if not runs:
        return "No faithfulness results yet.\n"
    lines = [
        "| Answers | Judge | Faithfulness | AUC vs humans | Accuracy vs humans |",
        "|---|---|---|---|---|",
    ]
    for dataset, r in runs:
        f, a = r["summary"]["faithfulness"], r.get("agreement") or {}
        auc = f"{a['auc_faithfulness']:.3f}" if a else "n/a"
        acc = f"{a['accuracy_all_supported']:.3f}" if a else "n/a"
        lines.append(
            f"| {dataset} ({r['summary']['scored']}) | {r['name']} | "
            f"{f['mean']:.3f} ({f['ci'][0]:.3f} to {f['ci'][1]:.3f}) | {auc} | {acc} |"
        )
    # How accuracy compares with always answering "faithful", per labelled dataset.
    base = {}
    for dataset, r in runs:
        a = r.get("agreement")
        if a and a.get("labelled"):
            base.setdefault(dataset, (a["faithful"], a["labelled"]))
    base_note = "".join(
        f" In {d}, {f} of {n} answers are labelled faithful, so calling every answer "
        f"faithful scores {f / n:.3f} accuracy."
        for d, (f, n) in sorted(base.items())
    )
    lines.append(
        "\nFaithfulness is the mean share of supported claims per answer, with a 95% "
        "bootstrap interval. AUC is the chance a faithful answer outscores an unfaithful "
        "one; accuracy calls an answer faithful when every claim is supported; both against "
        f"the human labels.{base_note} {_source([r['meta'] for _, r in runs])}.\n"
    )
    return "\n".join(lines)


def _replace(text: str, start: str, end: str, body: str) -> str:
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}{end}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    text = args.readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"{args.readme} has no {START} ... {END} markers", file=sys.stderr)
        return 2
    new = _replace(text, START, END, render(args.results))
    if F_START in new and F_END in new:
        new = _replace(new, F_START, F_END, render_faithfulness(args.results))
    if args.check:
        if new != text:
            print("README results are out of date: run python scripts/update_readme.py")
            return 1
        return 0
    args.readme.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
