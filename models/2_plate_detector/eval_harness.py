"""Corners-only plate detection eval harness with LOPO cross-validation.

Evaluates how well YOLO-predicted plate corners + gold tip positions map to
correct wells. Trains k-fold YOLO models (leave-one-plate-out), fits 2D
calibration on training plates, evaluates on held-out plate.

Saves model weights, overlay images, and metrics to a timestamped run directory.

Usage:
    python experiments/eval_corners_harness.py --description "baseline"
    python experiments/eval_corners_harness.py --fliplr 0.5 --sigma 0.05 --epochs 200
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_DIR / "pipette_well_dataset"
FRAMES_DIR = PROJECT_DIR / "experiments" / "frames"
CORNERS_PATH = DATASET_DIR / "plate_corners.json"
LABELLED_FRAMES_PATH = DATASET_DIR / "labelled_frames.json"
LABELS_PATH = DATASET_DIR / "labels.json"
RUNS_BASE = PROJECT_DIR / "experiments" / "corner_eval_runs"

KP_NAMES_CORNERS = ["A1", "A12", "H12", "H1"]
KP_NAMES_WITH_TIP = ["A1", "A12", "H12", "H1", "tip"]
KP_NAMES_WITH_TIP_NOZZLE = ["A1", "A12", "H12", "H1", "tip", "nozzle"]
KP_NAMES = KP_NAMES_CORNERS  # overridden by --with-tip / --with-nozzle

GRID_COORDS = {
    "A1": np.array([0.0, 0.0]),
    "A12": np.array([11.0, 0.0]),
    "H12": np.array([11.0, 7.0]),
    "H1": np.array([0.0, 7.0]),
}

COLORS_BGR = {
    "A1": (128, 222, 74),
    "A12": (250, 165, 96),
    "H12": (22, 115, 249),
    "H1": (252, 132, 192),
    "tip": (94, 63, 244),
    "nozzle": (0, 200, 255),
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load all three data files."""
    with open(CORNERS_PATH) as f:
        corners = json.load(f)
    with open(LABELLED_FRAMES_PATH) as f:
        labelled_frames = json.load(f)
    with open(LABELS_PATH) as f:
        labels_raw = json.load(f)

    labels_lookup = {}
    for entry in labels_raw:
        clip_id = entry["clip_id_FPV"].replace("_FPV", "")
        labels_lookup[clip_id] = entry["wells_ground_truth"]

    return corners, labelled_frames, labels_lookup


def infer_pipette_type_from_wells(wells):
    n = len(wells)
    if n == 1:
        return "single"
    elif n == 12:
        return "multi-col"
    elif n == 8:
        return "multi-row"
    return "single"


def filter_single_pipette(corners, labelled_frames, labels_lookup):
    """Filter to single-pipette clips only."""
    single_clips = set()
    for clip_id in corners:
        if clip_id in labels_lookup:
            ptype = infer_pipette_type_from_wells(labels_lookup[clip_id])
            if ptype == "single":
                single_clips.add(clip_id)

    corners = {k: v for k, v in corners.items() if k in single_clips}
    labelled_frames = {k: v for k, v in labelled_frames.items() if k in single_clips}
    return corners, labelled_frames


# ── YOLO Dataset Export ──────────────────────────────────────────────────────

def annotation_to_yolo_label(annotation, view="fpv"):
    """Convert one clip's annotation to a YOLO-pose label line."""
    view_data = annotation.get(view)
    if not view_data:
        return None

    ann_corners = view_data.get("corners", {})

    kps = []
    visible_xs = []
    visible_ys = []

    for name in KP_NAMES:
        if name in ("tip", "nozzle"):
            pt = view_data.get(name)
        else:
            pt = ann_corners.get(name)
        if pt is not None:
            kps.append((pt[0], pt[1], 2))
            visible_xs.append(pt[0])
            visible_ys.append(pt[1])
        else:
            kps.append((0.0, 0.0, 0))

    if len(visible_xs) < 3:
        return None

    min_x, max_x = min(visible_xs), max(visible_xs)
    min_y, max_y = min(visible_ys), max(visible_ys)
    w = max_x - min_x
    h = max_y - min_y
    pad_x = w * 0.10
    pad_y = h * 0.10
    min_x = max(0.0, min_x - pad_x)
    max_x = min(1.0, max_x + pad_x)
    min_y = max(0.0, min_y - pad_y)
    max_y = min(1.0, max_y + pad_y)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    bw = max_x - min_x
    bh = max_y - min_y

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for kx, ky, kv in kps:
        parts.append(f"{kx:.6f} {ky:.6f} {kv}")

    return " ".join(parts)


def export_fold_dataset(corners, labelled_frames, fold_dir, train_plates, val_plates,
                        nearby_frames=2, fliplr=0.0):
    """Export a single fold's YOLO-pose dataset with flip_idx support."""
    for split in ["train", "val"]:
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0}

    for clip_id, annotation in sorted(corners.items()):
        plate = clip_id.rsplit("_clip", 1)[0]

        if plate in val_plates:
            split = "val"
        elif plate in train_plates:
            split = "train"
        else:
            continue

        label_line = annotation_to_yolo_label(annotation, view="fpv")
        if label_line is None:
            stats["skipped"] += 1
            continue

        frame_idx = annotation["frame"]
        lf = labelled_frames.get(clip_id, {})
        start_frame = lf.get("start_frame", frame_idx)
        end_frame = lf.get("end_frame", frame_idx)

        offsets = [0] + list(range(-nearby_frames, 0)) + list(range(1, nearby_frames + 1))

        for offset in offsets:
            fidx = frame_idx + offset
            if fidx < start_frame or fidx > end_frame:
                continue

            frame_str = f"frame_{fidx:04d}.jpg"
            src_path = FRAMES_DIR / f"{clip_id}_FPV" / frame_str
            if not src_path.exists():
                continue

            suffix = f"_off{offset}" if offset != 0 else ""
            img_name = f"{clip_id}_fpv{suffix}.jpg"
            label_name = f"{clip_id}_fpv{suffix}.txt"

            shutil.copy2(src_path, fold_dir / "images" / split / img_name)
            with open(fold_dir / "labels" / split / label_name, "w") as f:
                f.write(label_line + "\n")

            stats[split] += 1

    # Write dataset.yaml
    yaml_lines = [
        f"path: {fold_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"kpt_shape: [{len(KP_NAMES)}, 3]",
    ]

    # Add flip_idx when horizontal flipping is enabled
    # Corners swap: A1↔A12 (0↔1), H1↔H12 (3↔2). Tip and nozzle map to themselves.
    if fliplr > 0:
        if len(KP_NAMES) == 6:
            yaml_lines.append("flip_idx: [1, 0, 3, 2, 4, 5]")
        elif len(KP_NAMES) == 5:
            yaml_lines.append("flip_idx: [1, 0, 3, 2, 4]")
        else:
            yaml_lines.append("flip_idx: [1, 0, 3, 2]")

    yaml_lines.extend([
        "",
        "names:",
        "  0: plate",
        "",
    ])

    yaml_path = fold_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"  [{fold_dir.name}] train={stats['train']}, val={stats['val']}, skipped={stats['skipped']}")
    return yaml_path


# ── Training ─────────────────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_fold_model(dataset_yaml, run_dir, epochs=200, imgsz=1280, fliplr=0.5,
                     sigma=0.05, pose_weight=12.0):
    """Train a YOLO-pose model for one fold with custom sigma."""
    import torch
    from ultralytics import YOLO

    device = get_device()
    model = YOLO("yolo11n-pose.pt")

    # Sigma monkey-patch via callback (criterion only exists after first batch starts)
    sigma_val = sigma
    nkpt = len(KP_NAMES)
    patched = [False]

    def patch_sigma(trainer):
        """Override default OKS sigmas with tighter values on first batch."""
        if patched[0]:
            return
        try:
            crit = trainer.model.criterion
            if crit is not None and hasattr(crit, "keypoint_loss"):
                old = crit.keypoint_loss.sigmas
                crit.keypoint_loss.sigmas = torch.full((nkpt,), sigma_val, device=trainer.device)
                print(f"    Patched keypoint sigmas to {sigma_val} (was {old.tolist()})")
                patched[0] = True
        except Exception as e:
            print(f"    Warning: could not patch sigmas: {e}")

    model.add_callback("on_train_batch_start", patch_sigma)

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        patience=20,
        device=device,
        save=True,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        # Augmentation
        fliplr=fliplr,
        flipud=0.0,
        mosaic=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Loss weights
        pose=pose_weight,
        kobj=1.0,
        verbose=True,
    )

    best_path = Path(run_dir) / "train" / "weights" / "best.pt"
    return best_path if best_path.exists() else None


# ── Homography + Well Prediction ─────────────────────────────────────────────

def compute_homography(corners_px):
    """Compute perspective transform from pixel corners to grid coords."""
    available = {k: v for k, v in corners_px.items() if k in GRID_COORDS and v is not None}

    if len(available) >= 4:
        src = np.array([available[k] for k in ["A1", "A12", "H12", "H1"]], dtype=np.float32)
        dst = np.array([GRID_COORDS[k] for k in ["A1", "A12", "H12", "H1"]], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    if len(available) == 3:
        all_corners = ["A1", "A12", "H12", "H1"]
        missing = [k for k in all_corners if k not in available][0]

        if missing == "A1":
            available["A1"] = available["A12"] + available["H1"] - available["H12"]
        elif missing == "A12":
            available["A12"] = available["A1"] + available["H12"] - available["H1"]
        elif missing == "H12":
            available["H12"] = available["A12"] + available["H1"] - available["A1"]
        elif missing == "H1":
            available["H1"] = available["A1"] + available["H12"] - available["A12"]

        src = np.array([available[k] for k in ["A1", "A12", "H12", "H1"]], dtype=np.float32)
        dst = np.array([GRID_COORDS[k] for k in ["A1", "A12", "H12", "H1"]], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    return None


def map_tip_to_grid(tip_px, H):
    """Map tip pixel coords through homography to continuous grid position."""
    if tip_px is None or H is None:
        return None
    pt = np.array([[[tip_px[0], tip_px[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def grid_to_well(gx, gy):
    """Convert continuous grid coords to (row_letter, col_number)."""
    col = int(max(0, min(11, round(gx)))) + 1
    row_idx = int(max(0, min(7, round(gy))))
    return chr(65 + row_idx), col


# ── YOLO Inference ───────────────────────────────────────────────────────────

def extract_keypoints(model, frame_path, conf_thresh=0.3):
    """Run YOLO and extract keypoints in pixel coords.

    Returns dict of keypoint name -> np.array([x, y]) for detected keypoints.
    For corners-only mode: {"A1": ..., "A12": ..., "H12": ..., "H1": ...}
    For with-tip mode: also includes "tip" if detected.
    """
    results = model(str(frame_path), verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None

    r = results[0]
    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    if confs[best_idx] < conf_thresh:
        return None

    kps = r.keypoints.data[best_idx].cpu().numpy()
    keypoints = {}
    for i, name in enumerate(KP_NAMES):
        x, y, c = kps[i]
        if c > conf_thresh:
            keypoints[name] = np.array([x, y])

    # Need at least 3 corners for homography
    n_corners = sum(1 for k in KP_NAMES_CORNERS if k in keypoints)
    return keypoints if n_corners >= 3 else None


# ── Calibration (2D Linear Regression) ───────────────────────────────────────

def fit_2d_calibration(raw_predictions):
    """Fit 2D linear regression: col = a*gx + b*gy + c, row = d*gx + e*gy + f.

    Args:
        raw_predictions: list of dicts with 'gx', 'gy', 'gt_col', 'gt_row'

    Returns:
        (col_coefs, row_coefs) each [coef_gx, coef_gy, intercept]
    """
    features, gt_cols, gt_rows = [], [], []
    for p in raw_predictions:
        features.append([p["gx"], p["gy"], 1.0])
        gt_cols.append(p["gt_col"])
        gt_rows.append(p["gt_row"])

    if len(features) < 5:
        return [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

    X = np.array(features)
    col_coefs, _, _, _ = np.linalg.lstsq(X, gt_cols, rcond=None)
    row_coefs, _, _, _ = np.linalg.lstsq(X, gt_rows, rcond=None)

    return col_coefs.tolist(), row_coefs.tolist()


def apply_calibration(gx, gy, col_coefs, row_coefs):
    """Apply 2D calibration correction."""
    corrected_gx = col_coefs[0] * gx + col_coefs[1] * gy + col_coefs[2]
    corrected_gy = row_coefs[0] * gx + row_coefs[1] * gy + row_coefs[2]
    return corrected_gx, corrected_gy


# ── Overlay Drawing ──────────────────────────────────────────────────────────

def draw_overlay(img, corners, tip_px, pred_well, gt_str, is_correct,
                 raw_grid=None, corrected_grid=None, pred_tip_px=None,
                 pred_nozzle_px=None, gold_nozzle_px=None):
    """Draw predicted corners, tip, and result on frame."""
    overlay = img.copy()

    # Draw polygon connecting corners
    pts = []
    for name in ["A1", "A12", "H12", "H1"]:
        if name in corners:
            pts.append([int(corners[name][0]), int(corners[name][1])])
    if len(pts) >= 3:
        cv2.polylines(overlay, [np.array(pts, np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw corner keypoints
    for name, pt in corners.items():
        px, py = int(pt[0]), int(pt[1])
        color = COLORS_BGR.get(name, (255, 255, 255))
        cv2.circle(overlay, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, name, (px + 12, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Draw gold tip (green)
    if tip_px is not None:
        tx, ty = int(tip_px[0]), int(tip_px[1])
        cv2.circle(overlay, (tx, ty), 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (tx, ty), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "tip(gold)", (tx + 12, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw predicted tip (pink/magenta)
    if pred_tip_px is not None:
        px, py = int(pred_tip_px[0]), int(pred_tip_px[1])
        cv2.circle(overlay, (px, py), 8, COLORS_BGR["tip"], -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "tip(pred)", (px + 12, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR["tip"], 2, cv2.LINE_AA)

    # Draw gold nozzle (cyan outline)
    if gold_nozzle_px is not None:
        nx, ny = int(gold_nozzle_px[0]), int(gold_nozzle_px[1])
        cv2.circle(overlay, (nx, ny), 10, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "nozzle(gold)", (nx + 14, ny + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw predicted nozzle (yellow/orange)
    if pred_nozzle_px is not None:
        nx, ny = int(pred_nozzle_px[0]), int(pred_nozzle_px[1])
        cv2.circle(overlay, (nx, ny), 8, COLORS_BGR["nozzle"], -1, cv2.LINE_AA)
        cv2.circle(overlay, (nx, ny), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "nozzle(pred)", (nx + 12, ny + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR["nozzle"], 2, cv2.LINE_AA)

    # Prediction text
    status = "CORRECT" if is_correct else "WRONG"
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.putText(overlay, f"Pred: {pred_well}  GT: {gt_str}  [{status}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Grid coords info
    y_offset = 60
    if raw_grid:
        cv2.putText(overlay, f"Raw grid: ({raw_grid[0]:.2f}, {raw_grid[1]:.2f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += 25
    if corrected_grid:
        cv2.putText(overlay, f"Corrected: ({corrected_grid[0]:.2f}, {corrected_grid[1]:.2f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


# ── Main Evaluation Pipeline ─────────────────────────────────────────────────

def run_single_config(corners, labelled_frames, labels_lookup, config, run_dir):
    """Run full LOPO eval for a single training config. Returns LOPO accuracy."""

    all_plates = sorted(set(cid.rsplit("_clip", 1)[0] for cid in corners))
    main_plates = [p for p in all_plates if sum(1 for c in corners if c.startswith(p)) >= 3]

    print(f"\nPlates: {all_plates}")
    print(f"Plates for k-fold (>=3 clips): {main_plates}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Save config
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    data_dir = run_dir / "yolo_data"

    # ── Phase 1: Export k-fold datasets ──────────────────────────────────
    print("\n=== EXPORTING K-FOLD DATASETS ===")
    for i, val_plate in enumerate(main_plates):
        train_plates = [p for p in all_plates if p != val_plate]
        fold_dir = data_dir / f"fold_{i}"
        print(f"\n  Fold {i}: val={val_plate}, train={train_plates}")
        export_fold_dataset(
            corners, labelled_frames, fold_dir,
            train_plates=train_plates, val_plates=[val_plate],
            nearby_frames=config.get("nearby_frames", 2),
            fliplr=config.get("fliplr", 0.0),
        )

    # ── Phase 2: Train k-fold models ────────────────────────────────────
    print("\n=== TRAINING K-FOLD MODELS ===")
    fold_models = []
    for i, val_plate in enumerate(main_plates):
        print(f"\n--- Fold {i}: val={val_plate} ---")
        yaml_path = data_dir / f"fold_{i}" / "dataset.yaml"
        fold_run_dir = data_dir / f"fold_{i}" / "runs"

        best_path = train_fold_model(
            yaml_path, fold_run_dir,
            epochs=config.get("epochs", 200),
            imgsz=config.get("imgsz", 1280),
            fliplr=config.get("fliplr", 0.5),
            sigma=config.get("sigma", 0.05),
            pose_weight=config.get("pose_weight", 12.0),
        )
        fold_models.append((val_plate, best_path))

        # Copy best weights to run output
        out_fold = run_dir / "fold_results" / f"fold_{i}_{val_plate}"
        (out_fold / "weights").mkdir(parents=True, exist_ok=True)
        if best_path and best_path.exists():
            shutil.copy2(best_path, out_fold / "weights" / "best.pt")

    # ── Phase 3: LOPO Evaluation ────────────────────────────────────────
    print("\n=== LOPO EVALUATION ===")
    from ultralytics import YOLO

    # Collect raw predictions across all folds (for calibration fitting)
    all_raw_preds = {}  # plate -> list of {clip_id, gx, gy, gt_col, gt_row}
    all_clip_details = {}  # clip_id -> full detail dict

    for i, (val_plate, model_path) in enumerate(fold_models):
        if model_path is None or not model_path.exists():
            print(f"  Fold {i} ({val_plate}): model not found, skipping")
            continue

        model = YOLO(str(model_path))

        # Get clips on this val plate
        val_clips = {cid: fr for cid, fr in labelled_frames.items()
                     if cid.startswith(val_plate)}

        # Also predict on training plates for calibration data
        train_clips = {cid: fr for cid, fr in labelled_frames.items()
                       if not cid.startswith(val_plate)}

        # Predict on all clips using this fold's model
        for clip_id, frame_range in sorted({**val_clips, **train_clips}.items()):
            mid_frame = (frame_range["start_frame"] + frame_range["end_frame"]) // 2
            frame_path = FRAMES_DIR / f"{clip_id}_FPV" / f"frame_{mid_frame:04d}.jpg"

            if not frame_path.exists():
                continue

            # Get gold tip from plate_corners.json
            ann = corners.get(clip_id, {})
            fpv_data = ann.get("fpv", {})
            gold_tip_norm = fpv_data.get("tip")
            if gold_tip_norm is None:
                continue

            # Scale normalized tip to pixel coords
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gold_tip_px = np.array([gold_tip_norm[0] * w, gold_tip_norm[1] * h])

            # Gold nozzle position (if available)
            gold_nozzle_norm = fpv_data.get("nozzle")
            gold_nozzle_px = np.array([gold_nozzle_norm[0] * w, gold_nozzle_norm[1] * h]) \
                if gold_nozzle_norm is not None else None

            # Predict keypoints (corners + optionally tip)
            pred_kps = extract_keypoints(model, frame_path)
            if pred_kps is None:
                continue

            # Separate corners from tip
            pred_corners = {k: v for k, v in pred_kps.items() if k in KP_NAMES_CORNERS}

            # Use predicted tip if available, otherwise gold tip
            use_tip = pred_kps.get("tip") if "tip" in KP_NAMES else None
            tip_for_mapping = use_tip if use_tip is not None else gold_tip_px
            tip_source = "predicted" if use_tip is not None else "gold"

            # Homography + map tip
            H = compute_homography(pred_corners)
            if H is None:
                continue

            grid = map_tip_to_grid(tip_for_mapping, H)
            if grid is None:
                continue

            gx, gy = grid

            # Ground truth
            gt_wells = labels_lookup.get(clip_id)
            if not gt_wells:
                continue
            gt_col = int(gt_wells[0]["well_column"]) - 1  # 0-indexed
            gt_row = ord(gt_wells[0]["well_row"]) - ord("A")  # 0-indexed

            plate = clip_id.rsplit("_clip", 1)[0]
            ptype = infer_pipette_type_from_wells(gt_wells)
            pred_entry = {
                "clip_id": clip_id,
                "gx": gx, "gy": gy,
                "gt_col": gt_col, "gt_row": gt_row,
                "gt_wells": gt_wells,
                "pipette_type": ptype,
                "plate": plate,
                "fold_idx": i,
                "is_val": clip_id in val_clips,
                "pred_corners": {k: v.tolist() for k, v in pred_corners.items()},
                "gold_tip_px": gold_tip_px.tolist(),
                "gold_nozzle_px": gold_nozzle_px.tolist() if gold_nozzle_px is not None else None,
                "pred_tip_px": use_tip.tolist() if use_tip is not None else None,
                "pred_nozzle_px": pred_kps["nozzle"].tolist() if "nozzle" in pred_kps else None,
                "tip_source": tip_source,
                "mid_frame": mid_frame,
                "frame_path": str(frame_path),
                "n_corners": len(pred_corners),
            }

            if plate not in all_raw_preds:
                all_raw_preds[plate] = []
            all_raw_preds[plate].append(pred_entry)
            all_clip_details[clip_id] = pred_entry

    # ── Phase 4: LOPO Calibration + Scoring ─────────────────────────────
    print("\n=== LOPO CALIBRATION ===")

    lopo_results = []
    total_correct = 0
    total_clips = 0
    total_correct_raw = 0
    # Track by pipette type
    ptype_correct = {"single": 0, "multi-col": 0, "multi-row": 0}
    ptype_total = {"single": 0, "multi-col": 0, "multi-row": 0}

    for i, val_plate in enumerate(main_plates):
        # Training data = all plates except val_plate
        train_preds = []
        for plate, preds in all_raw_preds.items():
            if plate != val_plate:
                train_preds.extend(preds)

        # Val data = val_plate predictions (from fold i model only)
        val_preds = [p for p in all_raw_preds.get(val_plate, []) if p["fold_idx"] == i]

        if not val_preds:
            print(f"  Fold {i} ({val_plate}): no val predictions, skipping")
            continue

        # Fit calibration on SINGLE-pipette training clips only
        # (multi-pipette clips have misleading GT: first well != tip position)
        single_train_preds = [p for p in train_preds if p.get("pipette_type", "single") == "single"]
        col_coefs, row_coefs = fit_2d_calibration(single_train_preds if single_train_preds else train_preds)

        # Apply to val plate
        fold_correct = 0
        fold_correct_raw = 0
        fold_total = 0
        fold_details = []

        out_fold = run_dir / "fold_results" / f"fold_{i}_{val_plate}"
        overlay_dir = out_fold / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for p in val_preds:
            gx, gy = p["gx"], p["gy"]
            gt_col, gt_row = p["gt_col"], p["gt_row"]
            ptype = p.get("pipette_type", "single")

            # Raw (uncalibrated) prediction
            raw_row, raw_col = grid_to_well(gx, gy)

            # Calibrated prediction
            cgx, cgy = apply_calibration(gx, gy, col_coefs, row_coefs)
            pred_row, pred_col = grid_to_well(cgx, cgy)
            pred_well = f"{pred_row}{pred_col}"

            gt_row_letter = chr(65 + gt_row)
            gt_col_num = gt_col + 1
            gt_str = f"{gt_row_letter}{gt_col_num}"

            # Multi-well-aware correctness:
            # - single: both row AND column must match
            # - multi-col: only ROW must match (all 12 columns filled)
            # - multi-row: only COLUMN must match (all 8 rows filled)
            if ptype == "multi-col":
                is_correct = (pred_row == gt_row_letter)
                is_correct_raw = (raw_row == gt_row_letter)
            elif ptype == "multi-row":
                is_correct = (pred_col == gt_col_num)
                is_correct_raw = (raw_col == gt_col_num)
            else:
                is_correct = (pred_row == gt_row_letter and pred_col == gt_col_num)
                is_correct_raw = (raw_row == chr(65 + gt_row) and raw_col == gt_col + 1)

            fold_total += 1
            ptype_total[ptype] += 1
            if is_correct:
                fold_correct += 1
                ptype_correct[ptype] += 1
            if is_correct_raw:
                fold_correct_raw += 1

            detail = {
                "clip_id": p["clip_id"],
                "raw_grid": [round(gx, 3), round(gy, 3)],
                "corrected_grid": [round(cgx, 3), round(cgy, 3)],
                "pred_well": pred_well,
                "gt_well": gt_str,
                "correct": is_correct,
                "correct_raw": is_correct_raw,
                "n_corners": p["n_corners"],
                "tip_source": p.get("tip_source", "gold"),
                "pipette_type": ptype,
            }
            fold_details.append(detail)

            # Draw overlay
            frame_path = p["frame_path"]
            img = cv2.imread(frame_path)
            if img is not None:
                pred_corners_np = {k: np.array(v) for k, v in p["pred_corners"].items()}
                gold_tip_px = np.array(p["gold_tip_px"])
                pred_tip_px = np.array(p["pred_tip_px"]) if p.get("pred_tip_px") else None
                pred_nozzle_px = np.array(p["pred_nozzle_px"]) if p.get("pred_nozzle_px") else None
                gold_nozzle_px = np.array(p["gold_nozzle_px"]) if p.get("gold_nozzle_px") else None
                overlay = draw_overlay(
                    img, pred_corners_np, gold_tip_px,
                    pred_well, gt_str, is_correct,
                    raw_grid=(gx, gy),
                    corrected_grid=(cgx, cgy),
                    pred_tip_px=pred_tip_px,
                    pred_nozzle_px=pred_nozzle_px,
                    gold_nozzle_px=gold_nozzle_px,
                )
                cv2.imwrite(str(overlay_dir / f"{p['clip_id']}.jpg"), overlay)

        total_correct += fold_correct
        total_correct_raw += fold_correct_raw
        total_clips += fold_total

        pct = fold_correct / fold_total * 100 if fold_total else 0
        pct_raw = fold_correct_raw / fold_total * 100 if fold_total else 0
        # Determine dominant pipette type for this fold
        fold_ptypes = [d.get("pipette_type", "single") for d in fold_details]
        ptype_str = ""
        if any(pt != "single" for pt in fold_ptypes):
            ptype_str = f" [{fold_ptypes[0]}]"
        print(f"  Fold {i} ({val_plate}): {fold_correct}/{fold_total} ({pct:.0f}%) "
              f"calibrated, {fold_correct_raw}/{fold_total} ({pct_raw:.0f}%) raw{ptype_str}")

        # Save fold predictions
        with open(out_fold / "predictions.json", "w") as f:
            json.dump(fold_details, f, indent=2)

        lopo_results.append({
            "fold": i,
            "val_plate": val_plate,
            "total": fold_total,
            "correct_calibrated": fold_correct,
            "correct_raw": fold_correct_raw,
            "accuracy_calibrated": pct,
            "accuracy_raw": pct_raw,
            "col_coefs": col_coefs,
            "row_coefs": row_coefs,
            "details": fold_details,
        })

    # ── Phase 5: Summary ────────────────────────────────────────────────
    lopo_pct = total_correct / total_clips * 100 if total_clips else 0
    lopo_pct_raw = total_correct_raw / total_clips * 100 if total_clips else 0

    print(f"\n{'=' * 60}")
    print(f"LOPO OVERALL: {total_correct}/{total_clips} ({lopo_pct:.1f}%) calibrated")
    print(f"LOPO RAW:     {total_correct_raw}/{total_clips} ({lopo_pct_raw:.1f}%) uncalibrated")
    for pt in ["single", "multi-col", "multi-row"]:
        if ptype_total[pt] > 0:
            pct_pt = ptype_correct[pt] / ptype_total[pt] * 100
            print(f"  {pt:>10s}: {ptype_correct[pt]}/{ptype_total[pt]} ({pct_pt:.1f}%)")
    print(f"{'=' * 60}")

    summary = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "lopo_accuracy_calibrated": lopo_pct,
        "lopo_accuracy_raw": lopo_pct_raw,
        "total_clips": total_clips,
        "correct_calibrated": total_correct,
        "correct_raw": total_correct_raw,
        "ptype_breakdown": {
            pt: {"correct": ptype_correct[pt], "total": ptype_total[pt]}
            for pt in ["single", "multi-col", "multi-row"] if ptype_total[pt] > 0
        },
        "fold_results": lopo_results,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable summary
    lines = [
        f"Corners-Only Eval Harness - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Description: {config.get('description', 'N/A')}",
        "",
        "Config:",
    ]
    for k, v in sorted(config.items()):
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        f"LOPO Accuracy (calibrated): {total_correct}/{total_clips} ({lopo_pct:.1f}%)",
        f"LOPO Accuracy (raw):        {total_correct_raw}/{total_clips} ({lopo_pct_raw:.1f}%)",
        "",
        "By pipette type:",
    ])
    for pt in ["single", "multi-col", "multi-row"]:
        if ptype_total[pt] > 0:
            pct_pt = ptype_correct[pt] / ptype_total[pt] * 100
            lines.append(f"  {pt}: {ptype_correct[pt]}/{ptype_total[pt]} ({pct_pt:.1f}%)")
    lines.extend([
        "",
        "Per-fold breakdown:",
    ])
    for r in lopo_results:
        lines.append(f"  Fold {r['fold']} ({r['val_plate']}): "
                      f"{r['correct_calibrated']}/{r['total']} ({r['accuracy_calibrated']:.0f}%) cal, "
                      f"{r['correct_raw']}/{r['total']} ({r['accuracy_raw']:.0f}%) raw")

    lines.extend(["", "Per-clip details:"])
    for r in lopo_results:
        for d in r["details"]:
            mark = "OK" if d["correct"] else "XX"
            lines.append(f"  [{mark}] {d['clip_id']}: pred={d['pred_well']} gt={d['gt_well']} "
                          f"raw=({d['raw_grid'][0]:.2f},{d['raw_grid'][1]:.2f}) "
                          f"cal=({d['corrected_grid'][0]:.2f},{d['corrected_grid'][1]:.2f}) "
                          f"corners={d['n_corners']}")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nRun saved to: {run_dir}")
    return lopo_pct


def main():
    global KP_NAMES
    parser = argparse.ArgumentParser(description="Plate detection eval harness")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--pose-weight", type=float, default=12.0)
    parser.add_argument("--nearby-frames", type=int, default=2)
    parser.add_argument("--description", type=str, default="default")
    parser.add_argument("--with-tip", action="store_true",
                        help="Train 5 keypoints (4 corners + tip) and use predicted tip")
    parser.add_argument("--with-nozzle", action="store_true",
                        help="Train 6 keypoints (4 corners + tip + nozzle) and use predicted tip")
    parser.add_argument("--no-auto-retry", action="store_true",
                        help="Disable auto-retry on poor results")
    parser.add_argument("--single-only", action="store_true",
                        help="Only use single-pipette clips (old behavior). Default: include all pipette types")
    args = parser.parse_args()

    # --with-nozzle implies --with-tip
    if args.with_nozzle:
        args.with_tip = True

    # Set keypoint mode
    if args.with_nozzle:
        KP_NAMES = KP_NAMES_WITH_TIP_NOZZLE
        flip_idx = [1, 0, 3, 2, 4, 5] if args.fliplr > 0 else None
    elif args.with_tip:
        KP_NAMES = KP_NAMES_WITH_TIP
        flip_idx = [1, 0, 3, 2, 4] if args.fliplr > 0 else None
    else:
        KP_NAMES = KP_NAMES_CORNERS
        flip_idx = [1, 0, 3, 2] if args.fliplr > 0 else None

    # Load and filter data
    corners, labelled_frames, labels_lookup = load_data()
    if args.single_only:
        corners, labelled_frames = filter_single_pipette(corners, labelled_frames, labels_lookup)

    # Filter to clips that have required keypoint annotations
    if args.with_nozzle:
        has_kps = {cid for cid, ann in corners.items()
                   if ann.get("fpv", {}).get("tip") is not None
                   and ann.get("fpv", {}).get("nozzle") is not None}
        corners = {k: v for k, v in corners.items() if k in has_kps}
        labelled_frames = {k: v for k, v in labelled_frames.items() if k in has_kps}
    elif args.with_tip:
        has_tip = {cid for cid, ann in corners.items()
                   if ann.get("fpv", {}).get("tip") is not None}
        corners = {k: v for k, v in corners.items() if k in has_tip}
        labelled_frames = {k: v for k, v in labelled_frames.items() if k in has_tip}

    mode_str = " (6kp: corners+tip+nozzle)" if args.with_nozzle else \
               " (5kp: corners+tip)" if args.with_tip else ""
    filter_str = "single-pipette" if args.single_only else "all-pipette-types"
    print(f"Loaded {len(corners)} clips ({filter_str}){mode_str}")

    # Build primary config
    config = {
        "description": args.description,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "fliplr": args.fliplr,
        "sigma": args.sigma,
        "pose_weight": args.pose_weight,
        "nearby_frames": args.nearby_frames,
        "model_arch": "yolo11n-pose",
        "kp_names": list(KP_NAMES),
        "with_tip": args.with_tip,
        "with_nozzle": args.with_nozzle,
        "flip_idx": flip_idx,
        "single_only": args.single_only,
    }

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_BASE / f"run_{timestamp}_{args.description}"

    accuracy = run_single_config(corners, labelled_frames, labels_lookup, config, run_dir)

    # Auto-retry with variations if accuracy is poor
    if accuracy < 50 and not args.no_auto_retry:
        print(f"\n{'=' * 60}")
        print(f"Accuracy {accuracy:.1f}% < 50%, trying parameter variations...")
        print(f"{'=' * 60}")

        variations = [
            {"sigma": 0.1, "description": f"{args.description}_sigma01"},
            {"fliplr": 0.0, "flip_idx": None, "description": f"{args.description}_noflip"},
            {"epochs": 150, "description": f"{args.description}_ep150"},
        ]

        for var in variations:
            var_config = {**config, **var}
            var_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            var_run_dir = RUNS_BASE / f"run_{var_ts}_{var_config['description']}"

            print(f"\n>>> Trying variation: {var}")
            var_acc = run_single_config(
                corners, labelled_frames, labels_lookup, var_config, var_run_dir
            )
            print(f">>> Variation accuracy: {var_acc:.1f}%")

    print("\nDone! Check experiments/corner_eval_runs/ for results.")


if __name__ == "__main__":
    main()
