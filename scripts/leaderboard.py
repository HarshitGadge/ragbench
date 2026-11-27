"""Write leaderboard.json, a small public summary of every committed result.

    python scripts/leaderboard.py            # rewrite leaderboard.json from results/
    python scripts/leaderboard.py --check    # fail if leaderboard.json is out of date

The file holds each run's means and 95% intervals, its paired comparisons, and
where it came from (version, commit, workflow run), without per-query scores.
Pages that show a leaderboard read it from the main branch, so a merged results
PR updates them with no other change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "ragbench.leaderboard/1"
REPO = "https://github.com/HarshitGadge/ragbench"


def _provenance(meta: dict) -> dict:
    return {k: meta.get(k) for k in ("ragbench", "git_sha", "ci_run", "created", "command")}


def build(results_dir: Path) -> dict:
    retrieval, comparisons, faithfulness = [], [], []
    for path in sorted(results_dir.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name.startswith("compare"):
            for row in data:
                comparisons.append(
                    {
                        "dataset": path.parent.name,
                        "baseline": row["baseline"],
                        "candidate": row["candidate"],
                        "metric": row["metric"],
                        "diff": row["diff"]["mean"],
                        "ci": row["diff"]["ci"],
                        "queries": row["queries"],
                        "significant": row["significant"],
                    }
                )
            continue
        retrieval.append(
            {
                "dataset": data["dataset"]["name"],
                "queries": data["dataset"]["evaluated_queries"],
                "documents": data["dataset"]["documents"],
                "run": data["name"],
                "retriever": data["meta"].get("retriever"),
                "params": data["meta"].get("params", {}),
                "metrics": {
                    m: {"mean": v["mean"], "ci": v["ci"]} for m, v in data["metrics"].items()
                },
                "latency_ms_p50": data["latency_ms"]["p50"],
                "source": _provenance(data["meta"]),
                "file": f"{REPO}/blob/main/{path.relative_to(results_dir.parent).as_posix()}",
            }
        )
    for path in sorted((results_dir / "faithfulness").glob("*/*.json")):
        if path.name == "about.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        faithfulness.append(
            {
                "answers": path.parent.name,
                "scored": data["summary"]["scored"],
                "claims": data["summary"]["claims"],
                "judge": data["name"],
                "judge_spec": data["judge"],
                "params": data["params"],
                "faithfulness": data["summary"]["faithfulness"],
                "agreement": data.get("agreement"),
                "source": _provenance(data["meta"]),
                "file": f"{REPO}/blob/main/{path.relative_to(results_dir.parent).as_posix()}",
            }
        )
    retrieval.sort(key=lambda r: (r["dataset"], -r["metrics"]["ndcg@10"]["mean"], r["run"]))
    return {
        "schema": SCHEMA,
        "repository": REPO,
        "retrieval": retrieval,
        "comparisons": comparisons,
        "faithfulness": faithfulness,
    }


def render(results_dir: Path) -> str:
    return json.dumps(build(results_dir), indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "leaderboard.json")
    args = parser.parse_args(argv)
    new = render(args.results)
    if args.check:
        old = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if new != old:
            print("leaderboard.json is out of date: run python scripts/leaderboard.py")
            return 1
        return 0
    args.out.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
