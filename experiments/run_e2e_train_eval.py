"""Evaluate final (all-data) models on all clips — training accuracy baseline.

Usage:
    python3 experiments/run_e2e_train_eval.py
    python3 experiments/run_e2e_train_eval.py --run-name my_eval
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.e2e_pipeline import EndToEndPipeline

DATASET_DIR = PROJECT_DIR / "pipette_well_dataset"
LABELS_PATH = DATASET_DIR / "labels.json"
OUTPUT_BASE = PROJECT_DIR / "experiments" / "e2e_runs"


def load_labels():
    with open(LABELS_PATH) as f:
        raw = json.load(f)
    lookup = {}
    for entry in raw:
        clip_id = entry["clip_id_FPV"].replace("_FPV", "")
        lookup[clip_id] = {
            "clip_id_FPV": entry["clip_id_FPV"],
            "clip_id_Topview": entry["clip_id_Topview"],
            "wells_ground_truth": entry["wells_ground_truth"],
        }
    return lookup


def wells_match(pred_wells, gt_wells):
    pred_set = {(w["well_row"], str(w["well_column"])) for w in pred_wells}
    gt_set = {(g["well_row"], str(g["well_column"])) for g in gt_wells}
    return pred_set == gt_set


def compute_errors(pred_wells, gt_wells):
    if not pred_wells or not gt_wells:
        return None, None
    pred = pred_wells[0]
    gt = gt_wells[0]
    pred_row = ord(pred["well_row"]) - ord("A")
    gt_row = ord(gt["well_row"]) - ord("A")
    pred_col = int(pred["well_column"]) - 1
    gt_col = int(gt["well_column"]) - 1
    return pred_row - gt_row, pred_col - gt_col


def infer_pipette_type_from_gt(gt_wells):
    n = len(gt_wells)
    if n == 1:
        return "single"
    elif n == 12:
        return "multi-col"
    elif n == 8:
        return "multi-row"
    return "single"


def within_n(pred_wells, gt_wells, n):
    if not pred_wells or not gt_wells:
        return False
    pred = pred_wells[0]
    gt = gt_wells[0]
    row_diff = abs(ord(pred["well_row"]) - ord(gt["well_row"]))
    col_diff = abs(int(pred["well_column"]) - int(gt["well_column"]))
    return max(row_diff, col_diff) <= n


def main():
    parser = argparse.ArgumentParser(description="E2E training accuracy eval (final models, all clips)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or "train_accuracy"
    output_dir = OUTPUT_BASE / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels()

    print(f"E2E Training Accuracy Evaluation (final models on all clips)")
    print(f"Output: {output_dir}")
    print(f"Total clips: {len(labels)}")
    print(f"{'=' * 60}")

    # Load pipeline once with final models
    t0 = time.time()
    pipeline = EndToEndPipeline(generate_overlays=True)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    all_clips = []
    total_correct = 0
    total_clips = 0
    total_row_ok = 0
    total_col_ok = 0
    total_within_1 = 0
    total_within_2 = 0
    type_correct = 0
    stage_fails = {"stage1": 0, "stage2": 0, "stage3": 0}
    plate_breakdown = {}

    for clip_id in sorted(labels.keys()):
        entry = labels[clip_id]
        fpv_path = str(DATASET_DIR / (entry["clip_id_FPV"] + ".mp4"))
        tv_path = str(DATASET_DIR / (entry["clip_id_Topview"] + ".mp4"))
        gt = entry["wells_ground_truth"]

        if not os.path.exists(fpv_path) or not os.path.exists(tv_path):
            print(f"  SKIP {clip_id} — video files not found")
            continue

        t0 = time.time()
        result = pipeline.predict(
            fpv_path, tv_path,
            ground_truth=gt,
            overlay_dir=str(output_dir),
        )
        elapsed = time.time() - t0

        pred_wells = result["wells_prediction"]
        correct = wells_match(pred_wells, gt)
        row_err, col_err = compute_errors(pred_wells, gt)
        gt_pipette_type = infer_pipette_type_from_gt(gt)
        pred_pipette_type = result.get("pipette_type", "single")
        pred_well_str = f"{pred_wells[0]['well_row']}{pred_wells[0]['well_column']}" if pred_wells else "?"
        gt_wells_str = [f"{g['well_row']}{g['well_column']}" for g in gt]

        status_icon = "OK" if correct else "MISS"
        if result["status"] != "ok":
            status_icon = result["status"].upper()

        type_match = "=" if pred_pipette_type == gt_pipette_type else "!"
        print(f"  [{status_icon:10s}] {clip_id:30s} "
              f"pred={pred_well_str:4s} gt={gt_wells_str[0]:4s} "
              f"({elapsed:.1f}s) {pred_pipette_type}{type_match}{gt_pipette_type}")

        plate = clip_id.split("_clip")[0]
        total_clips += 1
        if correct:
            total_correct += 1
        if pred_pipette_type == gt_pipette_type:
            type_correct += 1
        if row_err is not None and row_err == 0:
            total_row_ok += 1
        if col_err is not None and col_err == 0:
            total_col_ok += 1
        if within_n(pred_wells, gt, 1):
            total_within_1 += 1
        if within_n(pred_wells, gt, 2):
            total_within_2 += 1

        if result["status"] == "stage1_fail":
            stage_fails["stage1"] += 1
        elif result["status"] == "stage2_fail":
            stage_fails["stage2"] += 1
        elif result["status"] == "stage3_fail":
            stage_fails["stage3"] += 1

        if plate not in plate_breakdown:
            plate_breakdown[plate] = {"total": 0, "correct": 0}
        plate_breakdown[plate]["total"] += 1
        if correct:
            plate_breakdown[plate]["correct"] += 1

        clip_entry = {
            "clip_name": clip_id,
            "plate": plate,
            "status": result["status"],
            "correct": correct,
            "pred_well": pred_well_str,
            "gt_wells": gt_wells_str,
            "pipette_type": pred_pipette_type,
            "gt_pipette_type": gt_pipette_type,
            "commit_frame_idx": result["stages"].get("commit_frame", {}).get("frame_idx"),
            "n_corners_detected": result["stages"].get("plate_detection", {}).get("n_corners", 0),
            "pred_row": result["stages"].get("tip_detection", {}).get("pred_row"),
            "pred_col": result["stages"].get("tip_detection", {}).get("pred_col"),
            "row_err": row_err,
            "col_err": col_err,
            "elapsed": round(elapsed, 2),
            "overlays": {
                "commit": f"{clip_id}_s1_commit.jpg",
                "corners": f"{clip_id}_s2_corners.jpg",
                "warped": f"{clip_id}_s3_warped.jpg",
                "summary": f"{clip_id}_summary.jpg",
            },
        }
        all_clips.append(clip_entry)

    # Per-plate accuracy
    for plate in plate_breakdown:
        ps = plate_breakdown[plate]
        ps["accuracy_pct"] = round(100 * ps["correct"] / ps["total"], 1) if ps["total"] else 0

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING ACCURACY (final models on all clips)")
    print(f"{'=' * 60}")
    pct = lambda n: round(100 * n / total_clips, 1) if total_clips else 0
    print(f"Exact match:  {total_correct}/{total_clips} ({pct(total_correct)}%)")
    print(f"Row OK:       {total_row_ok}/{total_clips} ({pct(total_row_ok)}%)")
    print(f"Col OK:       {total_col_ok}/{total_clips} ({pct(total_col_ok)}%)")
    print(f"Within 1:     {total_within_1}/{total_clips} ({pct(total_within_1)}%)")
    print(f"Within 2:     {total_within_2}/{total_clips} ({pct(total_within_2)}%)")
    print(f"Type correct: {type_correct}/{total_clips} ({pct(type_correct)}%)")
    print(f"Stage fails:  S1={stage_fails['stage1']} S2={stage_fails['stage2']} S3={stage_fails['stage3']}")
    print(f"\nPer-plate:")
    for plate in sorted(plate_breakdown.keys(), key=lambda x: int(x.replace("Plate_", ""))):
        ps = plate_breakdown[plate]
        print(f"  {plate:12s}: {ps['correct']}/{ps['total']} ({ps['accuracy_pct']}%)")

    # Save results
    summary = {
        "total_clips": total_clips,
        "exact_match": total_correct,
        "exact_pct": pct(total_correct),
        "row_ok": total_row_ok,
        "row_ok_pct": pct(total_row_ok),
        "col_ok": total_col_ok,
        "col_ok_pct": pct(total_col_ok),
        "within_1": total_within_1,
        "within_1_pct": pct(total_within_1),
        "within_2": total_within_2,
        "within_2_pct": pct(total_within_2),
        "type_correct": type_correct,
        "type_correct_pct": pct(type_correct),
        "stage1_fails": stage_fails["stage1"],
        "stage2_fails": stage_fails["stage2"],
        "stage3_fails": stage_fails["stage3"],
        "plate_breakdown": plate_breakdown,
    }

    output = {"summary": summary, "clips": all_clips}
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
