"""Train final corners-only model on ALL plates (no held-out fold).

Uses the same training config as the LOPO run (200 epochs, imgsz=1280,
fliplr=0.5, sigma=0.05) but trains on all 100 clips.
"""
import shutil
from pathlib import Path
from eval_harness import (
    load_data, annotation_to_yolo_label, train_fold_model,
    KP_NAMES, RUNS_BASE, FRAMES_DIR,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent


def export_all_train_dataset(corners, labelled_frames, fold_dir, all_plates,
                              nearby_frames=2, fliplr=0.0):
    """Export dataset with ALL clips as train. Copies one plate to val for YOLO."""
    for split in ["train", "val"]:
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0}
    val_plate = all_plates[0]  # One plate duplicated into val for YOLO

    for clip_id, annotation in sorted(corners.items()):
        plate = clip_id.rsplit("_clip", 1)[0]
        if plate not in all_plates:
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

            # Everything goes to train
            shutil.copy2(src_path, fold_dir / "images" / "train" / img_name)
            with open(fold_dir / "labels" / "train" / label_name, "w") as f:
                f.write(label_line + "\n")
            stats["train"] += 1

            # Also copy one plate to val (YOLO requires val split)
            if plate == val_plate:
                shutil.copy2(src_path, fold_dir / "images" / "val" / img_name)
                with open(fold_dir / "labels" / "val" / label_name, "w") as f:
                    f.write(label_line + "\n")
                stats["val"] += 1

    # Write dataset.yaml
    yaml_lines = [
        f"path: {fold_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"kpt_shape: [{len(KP_NAMES)}, 3]",
    ]
    if fliplr > 0:
        yaml_lines.append("flip_idx: [1, 0, 3, 2]")
    yaml_lines.extend(["", "names:", "  0: plate", ""])

    yaml_path = fold_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"  train={stats['train']}, val={stats['val']}, skipped={stats['skipped']}")
    return yaml_path


def main():
    corners, labelled_frames, labels_lookup = load_data()

    all_plates = sorted(set(cid.rsplit("_clip", 1)[0] for cid in corners))
    main_plates = [p for p in all_plates if sum(1 for c in corners if c.startswith(p)) >= 3]

    print(f"Training on ALL plates: {main_plates}")
    print(f"Total clips: {len(corners)}")

    # Export dataset
    data_dir = RUNS_BASE / "final_all_plates" / "yolo_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = export_all_train_dataset(
        corners, labelled_frames, data_dir,
        all_plates=main_plates,
        nearby_frames=2,
        fliplr=0.5,
    )

    print(f"\nTraining final model...")

    run_dir = RUNS_BASE / "final_all_plates"
    train_fold_model(
        dataset_yaml=yaml_path,
        run_dir=run_dir,
        epochs=200,
        imgsz=1280,
        fliplr=0.5,
        sigma=0.05,
        pose_weight=12.0,
    )

    # Copy best.pt to models/
    best_pt = run_dir / "train" / "weights" / "best.pt"
    if best_pt.exists():
        dest = OUTPUT_DIR / "final.pt"
        shutil.copy2(best_pt, dest)
        print(f"\nFinal model saved to {dest}")
    else:
        print(f"\nWARNING: {best_pt} not found!")


if __name__ == "__main__":
    main()
