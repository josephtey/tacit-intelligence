"""Aggregate scores into a markdown report.

Usage:
    python -m eval.report --judge claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval.judge.base import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
SCORE_DIR = ROOT / "runs" / "scores"
REPORT_PATH = ROOT / "runs" / "report.md"


def load_scores(judge: str) -> dict[str, list[dict]]:
    """Returns generator_name -> list of score records."""
    out: dict[str, list[dict]] = {}
    judge_dir = SCORE_DIR / judge
    if not judge_dir.is_dir():
        return out
    for gen_dir in sorted(judge_dir.iterdir()):
        if not gen_dir.is_dir():
            continue
        records = []
        for fp in sorted(gen_dir.glob("*.json")):
            records.append(json.loads(fp.read_text()))
        out[gen_dir.name] = records
    return out


def agg(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude-opus-4-7")
    args = ap.parse_args()

    scores = load_scores(args.judge)
    if not scores:
        print(f"No scores found under {SCORE_DIR / args.judge}")
        return

    columns = DIMENSIONS + ["composite"]
    lines: list[str] = []
    lines.append(f"# Eval Report — judge: `{args.judge}`\n")
    lines.append(f"Generated from `{SCORE_DIR.relative_to(ROOT)}/{args.judge}/`.")
    lines.append("Each cell shows mean ± stdev across the manifest. Higher is better.\n")

    header = ["Model", "N"] + columns
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for gen_name, recs in sorted(scores.items()):
        row = [gen_name, str(len(recs))]
        for dim in DIMENSIONS:
            vals = [float(r["scores"][dim]["score"]) for r in recs if dim in r["scores"]]
            row.append(agg(vals))
        composite_vals = [float(r["composite"]) for r in recs if "composite" in r]
        row.append(agg(composite_vals))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Per-record composite, sorted (worst→best)\n")
    for gen_name, recs in sorted(scores.items()):
        lines.append(f"### {gen_name}\n")
        sorted_recs = sorted(recs, key=lambda r: r.get("composite", 0))
        lines.append("| slice_id | composite | summary |")
        lines.append("|---|---|---|")
        for r in sorted_recs:
            summary = r.get("summary", "").replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(f"| {r['slice_id']} | {r.get('composite', '—')} | {summary} |")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH.relative_to(ROOT)}")
    print()
    # Print the top table to stdout for quick inspection
    for line in lines[:8]:
        print(line)


if __name__ == "__main__":
    main()
