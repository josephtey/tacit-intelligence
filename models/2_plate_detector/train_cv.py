"""Train corners (plate detector) YOLO model using 5-fold stratified CV.

Reuses annotation_to_yolo_label and train_fold_model from eval_corners_harness.py,
but with clip-level stratified folds from cv_folds.json instead of LOPO.

Usage:
    python experiments/train_corners_cv_folds.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_DIR / "pipette_well_dataset"
FRAMES_DIR = PROJECT_DIR / "experiments" / "frames"
CORNERS_PATH = DATASET_DIR / "plate_corners.json"
LABELLED_FRAMES_PATH = DATASET_DIR / "labelled_frames.json"
LABELS_PATH = DATASET_DIR / "labels.json"
FOLDS_PATH = PROJECT_DIR / "experiments" / "cv_folds.json"
WORK_DIR = PROJECT_DIR / "experiments" / "corners_cv_data"
MODELS_DIR = Path(__file__).resolve().parent / "cv_folds"

# Same config as eval_corners_harness.py (4 keypoints, corners only)
KP_NAMES = ["A1", "A12", "H12", "H1"]
NEARBY_FRAMES = 2
EPOCHS = 200
IMGSZ = 1280
FLIPLR = 0.5
SIGMA = 0.05
POSE_WEIGHT = 12.0


def annotation_to_yolo_label(annotation, view="fpv"):
    """Convert one clip's annotation to a YOLO-pose label line (from harness)."""
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


def export_fold_dataset(corners, labelled_frames, fold_dir, train_clip_ids, val_clip_ids):
    """Export a single fold's YOLO-pose dataset using clip ID sets."""
    for split in ["train", "val"]:
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0}

    for clip_id, annotation in sorted(corners.items()):
        if clip_id in val_clip_ids:
            split = "val"
        elif clip_id in train_clip_ids:
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

        offsets = [0] + list(range(-NEARBY_FRAMES, 0)) + list(range(1, NEARBY_FRAMES + 1))

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
    if FLIPLR > 0:
        yaml_lines.append("flip_idx: [1, 0, 3, 2]")
    yaml_lines.extend(["", "names:", "  0: plate", ""])

    yaml_path = fold_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"  [{fold_dir.name}] train={stats['train']}, val={stats['val']}, skipped={stats['skipped']}")
    return yaml_path


def train_fold_model(dataset_yaml, run_dir):
    """Train a YOLO-pose model for one fold (same params as v4 corners)."""
    import torch
    from ultralytics import YOLO

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = YOLO("yolo11n-pose.pt")

    sigma_val = SIGMA
    nkpt = len(KP_NAMES)
    patched = [False]

    def patch_sigma(trainer):
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

    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=8,
        patience=20,
        device=device,
        save=True,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        fliplr=FLIPLR,
        flipud=0.0,
        mosaic=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        pose=POSE_WEIGHT,
        kobj=1.0,
        verbose=True,
    )

    best_path = Path(run_dir) / "train" / "weights" / "best.pt"
    return best_path if best_path.exists() else None


def main():
    # Load data
    with open(CORNERS_PATH) as f:
        corners = json.load(f)
    with open(LABELLED_FRAMES_PATH) as f:
        labelled_frames = json.load(f)
    with open(FOLDS_PATH) as f:
        folds_config = json.load(f)

    folds = folds_config["folds"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Corners Model: 5-Fold Stratified CV")
    print(f"Epochs={EPOCHS}, imgsz={IMGSZ}, fliplr={FLIPLR}, sigma={SIGMA}")
    print(f"{'='*60}")

    for fold in folds:
        fold_idx = fold["fold"]
        train_ids = set(fold["train_clips"])
        test_ids = set(fold["test_clips"])

        print(f"\n--- Fold {fold_idx}: {len(test_ids)} test, {len(train_ids)} train ---")

        # Export YOLO dataset
        fold_dir = WORK_DIR / f"fold_{fold_idx}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        yaml_path = export_fold_dataset(corners, labelled_frames, fold_dir, train_ids, test_ids)

        # Train
        run_dir = WORK_DIR / f"fold_{fold_idx}_run"
        best_path = train_fold_model(yaml_path, run_dir)

        if best_path and best_path.exists():
            dest = MODELS_DIR / f"fold_{fold_idx}.pt"
            shutil.copy2(best_path, dest)
            print(f"  Saved: {dest}")
        else:
            print(f"  WARNING: no best.pt found for fold {fold_idx}")

    # Train final model on ALL data
    print(f"\n{'='*60}")
    print("Training FINAL model on all clips...")
    print(f"{'='*60}")

    all_clip_ids = set()
    for fold in folds:
        all_clip_ids.update(fold["test_clips"])
        all_clip_ids.update(fold["train_clips"])

    # For final: all clips are train, use fold_0 test as dummy val (YOLO requires val)
    dummy_val = set(folds[0]["test_clips"])
    final_dir = WORK_DIR / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    yaml_path = export_fold_dataset(corners, labelled_frames, final_dir, all_clip_ids, dummy_val)

    run_dir = WORK_DIR / "final_run"
    best_path = train_fold_model(yaml_path, run_dir)

    if best_path and best_path.exists():
        dest = MODELS_DIR / "final.pt"
        shutil.copy2(best_path, dest)
        print(f"  Saved: {dest}")

    print(f"\n{'='*60}")
    print(f"All models saved to {MODELS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
