"""Run protocol generation across the manifest × generators.

Usage:
    python -m eval.run --models gpt-5 gemini-2.5-pro
    python -m eval.run --models gpt-5 --limit 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.generators import registry

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "runs" / "manifests" / "bench_50_seed42.json"
PROMPT_PATH = ROOT / "eval" / "prompts" / "generation.txt"
MAX_WORKERS_PER_PROVIDER = 4


def run_one(generator, record: dict, prompt: str) -> tuple[str, str, str | None]:
    """Returns (generator_name, slice_id, error_or_none)."""
    try:
        generator.generate(
            slice_id=record["slice_id"],
            video_path=record["video_path"],
            prompt=prompt,
        )
        return (generator.name, record["slice_id"], None)
    except Exception as e:
        return (generator.name, record["slice_id"], f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"Invalid shard {s!r}: need 0 <= i < n")
    return i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=registry.names(),
                    help=f"Models to run. Available: {registry.names()}")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run on only the first N records (smoke test).")
    ap.add_argument("--shard", default=None,
                    help='Process only records [i*N/n, (i+1)*N/n) where shard="i/n". '
                         'Used for SLURM array jobs.')
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
    prompt = PROMPT_PATH.read_text()
    print(f"Records: {len(records)}, models: {args.models}")
    print(f"Prompt: {len(prompt)} chars (sha1[:6]={__import__('hashlib').sha1(prompt.encode()).hexdigest()[:6]})\n")

    failures: list[tuple[str, str, str]] = []
    for model_name in args.models:
        gen = registry.get(model_name)
        print(f"\n=== Generating with {model_name} ===")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_PROVIDER) as pool:
            futures = [pool.submit(run_one, gen, r, prompt) for r in records]
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
    print("\nAll generations complete.")


if __name__ == "__main__":
    main()
