"""Unified evaluation harness: 3-stage pipeline vs zero-shot VLM baselines.

Runs one method over the dataset through the SAME scorer (imported from
run_e2e_lopo_eval) so accuracy is apples-to-apples, and aggregates per-stage
timing + token cost.

  --method e2e         3-stage pipeline, LOPO (fold-specific models per plate)
  --method vlm-mono    Gemini monolithic (1 call), zero-shot, all clips
  --method vlm-decomp  Gemini decomposed (2 calls), zero-shot, all clips

Examples:
    python3 experiments/run_baseline_eval.py --method e2e
    python3 experiments/run_baseline_eval.py --method vlm-mono --n-frames 16 --workers 8
    python3 experiments/run_baseline_eval.py --method vlm-decomp --limit 5   # quick test
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from experiments.run_e2e_lopo_eval import (  # reuse the exact same scorer
    load_labels, wells_match, compute_errors, within_n, infer_pipette_type_from_gt,
)
from src.e2e_pipeline import PLATES_SORTED

DATASET_DIR = PROJECT_DIR / "pipette_well_dataset"
OUTPUT_BASE = PROJECT_DIR / "experiments" / "baseline_runs"


def plate_of(clip_id):
    return clip_id.split("_clip")[0]


def evaluate_one(clip_id, entry, result):
    """Score one prediction result against ground truth."""
    gt = entry["wells_ground_truth"]
    pred_wells = result["wells_prediction"]
    pred_str = (f"{pred_wells[0]['well_row']}{pred_wells[0]['well_column']}"
                if pred_wells else "?")
    row_err, col_err = compute_errors(pred_wells, gt)
    return {
        "clip_name": clip_id,
        "plate": plate_of(clip_id),
        "status": result["status"],
        "correct": wells_match(pred_wells, gt),
        "pred_well": pred_str,
        "gt_wells": [f"{g['well_row']}{g['well_column']}" for g in gt],
        "pipette_type": result.get("pipette_type", "single"),
        "gt_pipette_type": infer_pipette_type_from_gt(gt),
        "row_err": row_err,
        "col_err": col_err,
        "within_1": within_n(pred_wells, gt, 1),
        "within_2": within_n(pred_wells, gt, 2),
        "timing": result.get("timing", {}),
        "cost": result.get("cost", {}),
        "stages": result.get("stages", {}),
    }


# ─────────────────────────── runners ───────────────────────────

def run_e2e(labels, plates, limit):
    """3-stage LOPO: load fold models per plate, predict serially."""
    from src.e2e_pipeline import EndToEndPipeline
    device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
    rows = []
    for fold_plate in plates:
        plate_clips = {k: v for k, v in labels.items()
                       if k.startswith(fold_plate + "_clip")}
        if not plate_clips:
            continue
        print(f"\nFold: hold out {fold_plate} ({len(plate_clips)} clips)")
        try:
            pipe = EndToEndPipeline(fold_plate=fold_plate, generate_overlays=False,
                                    device=device)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR loading {fold_plate}: {e}")
            continue
        for i, (clip_id, entry) in enumerate(sorted(plate_clips.items())):
            if limit and i >= limit:
                break
            fpv = str(DATASET_DIR / (entry["clip_id_FPV"] + ".mp4"))
            tv = str(DATASET_DIR / (entry["clip_id_Topview"] + ".mp4"))
            r = pipe.predict(fpv, tv, ground_truth=entry["wells_ground_truth"])
            row = evaluate_one(clip_id, entry, r)
            rows.append(row)
            print(f"  [{'OK' if row['correct'] else row['status']:9s}] "
                  f"{clip_id:24s} pred={row['pred_well']:4s} gt={row['gt_wells'][0]:4s} "
                  f"({r['timing'].get('total', 0):.1f}s)")
    return rows


def run_vlm(method, labels, limit, n_frames, workers):
    """Zero-shot VLM: one pipeline, all clips, concurrent (no folds — no training)."""
    from src.vlm_baseline import VLMMonolithicPipeline, VLMDecomposedPipeline
    Cls = VLMMonolithicPipeline if method == "vlm-mono" else VLMDecomposedPipeline
    pipe = Cls(n_frames=n_frames)

    items = sorted(labels.items())
    if limit:
        items = items[:limit]

    def task(args):
        clip_id, entry = args
        fpv = str(DATASET_DIR / (entry["clip_id_FPV"] + ".mp4"))
        tv = str(DATASET_DIR / (entry["clip_id_Topview"] + ".mp4"))
        r = pipe.predict(fpv, tv, ground_truth=entry["wells_ground_truth"])
        return evaluate_one(clip_id, entry, r)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(task, it): it[0] for it in items}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            print(f"  [{done:3d}/{len(items)}] [{'OK' if row['correct'] else row['status']:9s}] "
                  f"{row['clip_name']:24s} pred={row['pred_well']:4s} gt={row['gt_wells'][0]:4s} "
                  f"(${row['cost'].get('usd', 0):.3f}, {row['timing'].get('total', 0):.0f}s)")
    rows.sort(key=lambda r: r["clip_name"])
    return rows


# ─────────────────────────── aggregation ───────────────────────────

def summarize(rows, wall_clock):
    n = len(rows)
    pct = lambda c: round(100 * c / n, 1) if n else 0
    exact = sum(r["correct"] for r in rows)
    row_ok = sum(1 for r in rows if r["row_err"] == 0)
    col_ok = sum(1 for r in rows if r["col_err"] == 0)
    w1 = sum(1 for r in rows if r["within_1"])
    w2 = sum(1 for r in rows if r["within_2"])

    # per-plate
    by_plate = defaultdict(lambda: [0, 0])
    for r in rows:
        by_plate[r["plate"]][0] += int(r["correct"])
        by_plate[r["plate"]][1] += 1
    plate_breakdown = {p: {"correct": c, "total": t,
                           "accuracy_pct": round(100 * c / t, 1) if t else 0}
                       for p, (c, t) in by_plate.items()}

    # per pipette type (stratified failure modes)
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["gt_pipette_type"]][0] += int(r["correct"])
        by_type[r["gt_pipette_type"]][1] += 1
    type_breakdown = {t: {"correct": c, "total": tot,
                          "accuracy_pct": round(100 * c / tot, 1) if tot else 0}
                      for t, (c, tot) in by_type.items()}

    # timing: mean per stage key present
    timing_sum = defaultdict(float)
    timing_cnt = defaultdict(int)
    for r in rows:
        for k, v in r["timing"].items():
            timing_sum[k] += v
            timing_cnt[k] += 1
    timing_mean = {k: round(timing_sum[k] / timing_cnt[k], 2) for k in timing_sum}

    # cost totals
    cost_total = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "usd": 0.0}
    for r in rows:
        for k in cost_total:
            cost_total[k] += r["cost"].get(k, 0)
    cost_total["usd"] = round(cost_total["usd"], 4)
    cost_per_clip = round(cost_total["usd"] / n, 5) if n else 0

    return {
        "total_clips": n,
        "exact_match": exact, "exact_pct": pct(exact),
        "row_ok": row_ok, "row_ok_pct": pct(row_ok),
        "col_ok": col_ok, "col_ok_pct": pct(col_ok),
        "within_1": w1, "within_1_pct": pct(w1),
        "within_2": w2, "within_2_pct": pct(w2),
        "plate_breakdown": plate_breakdown,
        "type_breakdown": type_breakdown,
        "timing_mean_s": timing_mean,
        "total_cost": cost_total,
        "cost_per_clip_usd": cost_per_clip,
        "wall_clock_s": round(wall_clock, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["e2e", "vlm-mono", "vlm-decomp"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--plates", nargs="*", default=None, help="(e2e only)")
    ap.add_argument("--limit", type=int, default=None, help="cap clips (testing)")
    ap.add_argument("--n-frames", type=int, default=16, help="(vlm only)")
    ap.add_argument("--workers", type=int, default=8, help="(vlm only) concurrency")
    args = ap.parse_args()

    run_name = args.run_name or datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.method}")
    out_dir = OUTPUT_BASE / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    print(f"Method: {args.method}  |  clips: {len(labels)}  |  out: {out_dir}")
    print("=" * 64)

    t0 = time.time()
    if args.method == "e2e":
        rows = run_e2e(labels, args.plates or PLATES_SORTED, args.limit)
    else:
        rows = run_vlm(args.method, labels, args.limit, args.n_frames, args.workers)
    wall = time.time() - t0

    summary = summarize(rows, wall)

    print(f"\n{'=' * 64}\nRESULTS — {args.method}\n{'=' * 64}")
    print(f"Exact: {summary['exact_match']}/{summary['total_clips']} ({summary['exact_pct']}%)  "
          f"| row {summary['row_ok_pct']}%  col {summary['col_ok_pct']}%  "
          f"| w1 {summary['within_1_pct']}%  w2 {summary['within_2_pct']}%")
    print(f"By type: " + "  ".join(
        f"{t}={v['correct']}/{v['total']}" for t, v in summary["type_breakdown"].items()))
    print(f"Timing (mean s): {summary['timing_mean_s']}")
    print(f"Cost: ${summary['total_cost']['usd']} total  (${summary['cost_per_clip_usd']}/clip)  "
          f"| wall-clock {summary['wall_clock_s']}s")
    print(f"Per-plate: " + "  ".join(
        f"{p}={summary['plate_breakdown'][p]['accuracy_pct']}%"
        for p in PLATES_SORTED if p in summary["plate_breakdown"]))

    results = {
        "run_name": run_name, "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "config": {"n_frames": args.n_frames if args.method != "e2e" else None,
                   "workers": args.workers if args.method != "e2e" else None},
        "summary": summary, "clips": rows,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
