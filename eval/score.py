"""Score predictions against gold protocols using a judge model.

Usage:
    python -m eval.score --judge claude-opus-4-7
    python -m eval.score --judge claude-opus-4-7 --models gpt-5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.generators import registry as gen_registry
from eval.judge.claude import ClaudeJudge

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "runs" / "manifests" / "bench_50_seed42.json"
PRED_DIR = ROOT / "runs" / "predictions"
RUBRIC_PATH = ROOT / "eval" / "prompts" / "rubric.txt"
MAX_JUDGE_WORKERS = 4

JUDGES = {"claude-opus-4-7": ClaudeJudge}


def find_prediction(generator_name: str, slice_id: str) -> dict | None:
    candidates = sorted((PRED_DIR / generator_name).glob(f"{slice_id}__*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def score_one(judge, record, prediction, gold_text, rubric):
    try:
        judge.score(
            slice_id=record["slice_id"],
            generator_name=prediction["model"],
            predicted_protocol=prediction["protocol"],
            gold_protocol=gold_text,
            rubric_template=rubric,
        )
        return (prediction["model"], record["slice_id"], None)
    except Exception as e:
        return (prediction["model"], record["slice_id"], f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"Invalid shard {s!r}: need 0 <= i < n")
    return i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude-opus-4-7", choices=list(JUDGES))
    ap.add_argument("--models", nargs="+", default=gen_registry.names())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help='Process only records [i*N/n, (i+1)*N/n) where shard="i/n".')
    args = ap.parse_args()

    records = json.loads(MANIFEST.read_text())
    if args.limit:
        records = records[: args.limit]
    if args.shard:
        i, n = parse_shard(args.shard)
        N = len(records)
        start, end = (i * N) // n, ((i + 1) * N) // n
        records = records[start:end]
        print(f"Shard {i}/{n}: records [{start}:{end}] = {len(records)} records")
    rubric = RUBRIC_PATH.read_text()

    judge = JUDGES[args.judge]()
    print(f"Judge: {args.judge}")
    print(f"Records: {len(records)}, generators: {args.models}\n")

    failures: list[tuple[str, str, str]] = []
    for gen_name in args.models:
        print(f"\n=== Scoring {gen_name} ===")
        tasks = []
        for r in records:
            pred = find_prediction(gen_name, r["slice_id"])
            if pred is None:
                print(f"  SKIP {r['slice_id']}: no prediction for {gen_name}")
                continue
            gold = (ROOT / r["protocol_path"]).read_text(errors="replace")
            tasks.append((r, pred, gold))

        with ThreadPoolExecutor(max_workers=MAX_JUDGE_WORKERS) as pool:
            futures = [pool.submit(score_one, judge, r, p, g, rubric) for r, p, g in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                name, sid, err = fut.result()
                status = "OK" if err is None else "FAIL"
                print(f"  [{i:3d}/{len(futures)}] {name} {sid:14s} {status}")
                if err:
                    failures.append((name, sid, err))

    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for name, sid, err in failures:
            print(f"\n--- {name} | {sid} ---\n{err[:500]}", file=sys.stderr)
        sys.exit(1)
    print("\nAll scoring complete.")


if __name__ == "__main__":
    main()
