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
    ci_runs = sorted({r.meta.get("ci_run") for r in runs if r.meta.get("ci_run")})
    shas = sorted({r.meta.get("git_sha")[:7] for r in runs if r.meta.get("git_sha")})
    versions = sorted({r.meta.get("ragbench") for r in runs})
    source = f"ragbench {', '.join(versions)}"
    if shas:
        source += f", commit {', '.join(shas)}"
    if ci_runs:
        source += ", workflow run " + ", ".join(f"[{u.rsplit('/', 1)[-1]}]({u})" for u in ci_runs)
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
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render(args.results)}{END}{tail}"
    if args.check:
        if new != text:
            print("README results are out of date: run python scripts/update_readme.py")
            return 1
        return 0
    args.readme.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
