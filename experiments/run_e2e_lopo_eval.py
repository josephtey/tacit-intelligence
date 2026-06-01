"""End-to-end LOPO evaluation across all 7 plates.

For each plate, loads fold-specific models (RF, YOLO, HeatNet) trained
without that plate, runs the full 3-stage pipeline on all clips from
the held-out plate, and saves results + overlay images.

Usage:
    python3 experiments/run_e2e_lopo_eval.py
    python3 experiments/run_e2e_lopo_eval.py --run-name my_eval
    python3 experiments/run_e2e_lopo_eval.py --plates Plate_1 Plate_2
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

from src.e2e_pipeline import EndToEndPipeline, PLATES_SORTED

DATASET_DIR = PROJECT_DIR / "pipette_well_dataset"
LABELS_PATH = DATASET_DIR / "labels.json"
OUTPUT_BASE = PROJECT_DIR / "experiments" / "e2e_runs"


def load_labels():
    """Load ground truth labels, keyed by clip_id (without _FPV/_Topview)."""
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
    """Check if predicted wells exactly match ground truth."""
    pred_set = {(w["well_row"], str(w["well_column"])) for w in pred_wells}
    gt_set = {(g["well_row"], str(g["well_column"])) for g in gt_wells}
    return pred_set == gt_set


def compute_errors(pred_wells, gt_wells):
    """Compute row/col errors for single-well predictions."""
    if not pred_wells or not gt_wells:
        return None, None

    # For multi-well, compare the representative well (first predicted)
    pred = pred_wells[0]
    gt = gt_wells[0]

    pred_row = ord(pred["well_row"]) - ord("A")
    gt_row = ord(gt["well_row"]) - ord("A")
    pred_col = int(pred["well_column"]) - 1
    gt_col = int(gt["well_column"]) - 1

    return pred_row - gt_row, pred_col - gt_col


def infer_pipette_type_from_gt(gt_wells):
    """Infer pipette type from ground truth well count."""
    n = len(gt_wells)
    if n == 1:
        return "single"
    elif n == 12:
        return "multi-col"
    elif n == 8:
        return "multi-row"
    return "single"


def within_n(pred_wells, gt_wells, n):
    """Check if prediction is within n wells of ground truth."""
    if not pred_wells or not gt_wells:
        return False
    pred = pred_wells[0]
    gt = gt_wells[0]
    row_diff = abs(ord(pred["well_row"]) - ord(gt["well_row"]))
    col_diff = abs(int(pred["well_column"]) - int(gt["well_column"]))
    return max(row_diff, col_diff) <= n


def main():
    parser = argparse.ArgumentParser(description="E2E LOPO evaluation")
    parser.add_argument("--run-name", default=None,
                        help="Name for this run (default: timestamped)")
    parser.add_argument("--plates", nargs="*", default=None,
                        help="Specific plates to evaluate (default: all)")
    args = parser.parse_args()

    # Setup
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_e2e_lopo")
    output_dir = OUTPUT_BASE / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plates = args.plates or PLATES_SORTED
    labels = load_labels()

    print(f"E2E LOPO Evaluation")
    print(f"Output: {output_dir}")
    print(f"Plates: {plates}")
    print(f"Total clips in dataset: {len(labels)}")
    print(f"{'=' * 60}")

    all_clips = []
    total_correct = 0
    total_clips = 0
    total_row_ok = 0
    total_col_ok = 0
    total_within_1 = 0
    total_within_2 = 0
    stage_fails = {"stage1": 0, "stage2": 0, "stage3": 0}
    plate_breakdown = {}

    for fold_plate in plates:
        print(f"\nFold: hold out {fold_plate}")
        print(f"-" * 40)

        # Get clips for this plate
        plate_clips = {k: v for k, v in labels.items()
                       if k.startswith(fold_plate + "_clip")}
        if not plate_clips:
            print(f"  No clips for {fold_plate}, skipping")
            continue

        # Load fold-specific pipeline
        t0 = time.time()
        try:
            pipeline = EndToEndPipeline(
                fold_plate=fold_plate,
                generate_overlays=True,
            )
        except Exception as e:
            print(f"  ERROR loading models for {fold_plate}: {e}")
            continue
        load_time = time.time() - t0
        print(f"  Models loaded in {load_time:.1f}s")

        plate_correct = 0
        plate_total = 0

        for clip_id, entry in sorted(plate_clips.items()):
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

            # Evaluate
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

            # Track stats
            plate_total += 1
            total_clips += 1
            if correct:
                plate_correct += 1
                total_correct += 1

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

            # Build clip entry
            clip_entry = {
                "clip_name": clip_id,
                "plate": fold_plate,
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

        plate_pct = round(100 * plate_correct / plate_total, 1) if plate_total else 0
        plate_breakdown[fold_plate] = {
            "total": plate_total,
            "correct": plate_correct,
            "accuracy_pct": plate_pct,
        }
        print(f"  {fold_plate}: {plate_correct}/{plate_total} ({plate_pct}%)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"OVERALL RESULTS")
    print(f"{'=' * 60}")
    overall_pct = round(100 * total_correct / total_clips, 1) if total_clips else 0
    print(f"Exact match: {total_correct}/{total_clips} ({overall_pct}%)")
    print(f"Row OK:      {total_row_ok}/{total_clips} ({round(100*total_row_ok/total_clips, 1) if total_clips else 0}%)")
    print(f"Col OK:      {total_col_ok}/{total_clips} ({round(100*total_col_ok/total_clips, 1) if total_clips else 0}%)")
    print(f"Within 1:    {total_within_1}/{total_clips} ({round(100*total_within_1/total_clips, 1) if total_clips else 0}%)")
    print(f"Within 2:    {total_within_2}/{total_clips} ({round(100*total_within_2/total_clips, 1) if total_clips else 0}%)")
    print(f"Stage fails: S1={stage_fails['stage1']} S2={stage_fails['stage2']} S3={stage_fails['stage3']}")
    print(f"\nPer-plate:")
    for plate in PLATES_SORTED:
        if plate in plate_breakdown:
            pb = plate_breakdown[plate]
            print(f"  {plate:12s}: {pb['correct']}/{pb['total']} ({pb['accuracy_pct']}%)")

    # Save results
    results = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_clips": total_clips,
            "exact_match": total_correct,
            "exact_pct": overall_pct,
            "row_ok": total_row_ok,
            "row_ok_pct": round(100 * total_row_ok / total_clips, 1) if total_clips else 0,
            "col_ok": total_col_ok,
            "col_ok_pct": round(100 * total_col_ok / total_clips, 1) if total_clips else 0,
            "within_1": total_within_1,
            "within_1_pct": round(100 * total_within_1 / total_clips, 1) if total_clips else 0,
            "within_2": total_within_2,
            "within_2_pct": round(100 * total_within_2 / total_clips, 1) if total_clips else 0,
            "stage1_fails": stage_fails["stage1"],
            "stage2_fails": stage_fails["stage2"],
            "stage3_fails": stage_fails["stage3"],
            "failed": stage_fails["stage1"] + stage_fails["stage2"] + stage_fails["stage3"],
            "plate_breakdown": plate_breakdown,
        },
        "clips": all_clips,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
