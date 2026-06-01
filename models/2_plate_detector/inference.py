"""Inference script for v4 all-plates corners model.

Detects 4 plate corner keypoints (A1, A12, H12, H1) from a first-person-view
frame, computes a homography to map a known tip position to grid coordinates,
and predicts the well.

Usage:
    # Single frame with known tip position (normalized 0-1 coords)
    python models/v4_all_plates_corners/inference.py \
        --frame path/to/frame.jpg \
        --tip 0.45 0.62

    # Single frame with tip in pixel coords
    python models/v4_all_plates_corners/inference.py \
        --frame path/to/frame.jpg \
        --tip-px 580 410

    # Show overlay visualization
    python models/v4_all_plates_corners/inference.py \
        --frame path/to/frame.jpg \
        --tip 0.45 0.62 \
        --show
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

KP_NAMES = ["A1", "A12", "H12", "H1"]
GRID_COORDS = {
    "A1": np.array([0.0, 0.0]),
    "A12": np.array([11.0, 0.0]),
    "H12": np.array([11.0, 7.0]),
    "H1": np.array([0.0, 7.0]),
}


def load_model(model_path=None):
    """Load the YOLO-Pose model."""
    from ultralytics import YOLO
    if model_path is None:
        model_path = MODEL_DIR / "final.pt"
    return YOLO(str(model_path))


def extract_corners(model, frame_path, conf_thresh=0.3):
    """Run YOLO inference and extract corner keypoints in pixel coords.

    Returns:
        dict of corner_name -> np.array([x, y]), or None if detection failed.
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
    corners = {}
    for i, name in enumerate(KP_NAMES):
        x, y, c = kps[i]
        if c > conf_thresh:
            corners[name] = np.array([x, y])

    n_detected = len(corners)
    if n_detected < 3:
        return None
    return corners


def compute_homography(corners_px):
    """Compute perspective transform from pixel corners to grid coords.

    Works with 3 or 4 corners. With 3, estimates the 4th via parallelogram.
    """
    available = {k: v for k, v in corners_px.items() if k in GRID_COORDS}

    if len(available) >= 4:
        src = np.array([available[k] for k in KP_NAMES], dtype=np.float32)
        dst = np.array([GRID_COORDS[k] for k in KP_NAMES], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    if len(available) == 3:
        missing = [k for k in KP_NAMES if k not in available][0]
        if missing == "A1":
            available["A1"] = available["A12"] + available["H1"] - available["H12"]
        elif missing == "A12":
            available["A12"] = available["A1"] + available["H12"] - available["H1"]
        elif missing == "H12":
            available["H12"] = available["A12"] + available["H1"] - available["A1"]
        elif missing == "H1":
            available["H1"] = available["A1"] + available["H12"] - available["A12"]
        src = np.array([available[k] for k in KP_NAMES], dtype=np.float32)
        dst = np.array([GRID_COORDS[k] for k in KP_NAMES], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    return None


def map_tip_to_well(tip_px, H):
    """Map tip pixel position through homography to a well prediction.

    Returns:
        (row_letter, col_number, grid_x, grid_y) e.g. ("A", 1, 0.12, 0.05)
    """
    pt = np.array([[[tip_px[0], tip_px[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    gx, gy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    col = int(max(0, min(11, round(gx)))) + 1
    row_idx = int(max(0, min(7, round(gy))))
    row_letter = chr(65 + row_idx)

    return row_letter, col, gx, gy


def predict_well(model, frame_path, tip_norm=None, tip_px=None):
    """Full pipeline: detect corners, compute homography, predict well.

    Args:
        model: loaded YOLO model
        frame_path: path to FPV frame image
        tip_norm: (x, y) normalized tip position (0-1), or
        tip_px: (x, y) tip position in pixels

    Returns:
        dict with prediction results, or None if detection failed.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Get tip in pixels
    if tip_px is not None:
        tip = np.array(tip_px, dtype=np.float64)
    elif tip_norm is not None:
        tip = np.array([tip_norm[0] * w, tip_norm[1] * h], dtype=np.float64)
    else:
        return None

    # Detect corners
    corners = extract_corners(model, frame_path)
    if corners is None:
        return None

    # Compute homography
    H = compute_homography(corners)
    if H is None:
        return None

    # Map tip to well
    row, col, gx, gy = map_tip_to_well(tip, H)

    return {
        "well_row": row,
        "well_column": col,
        "well": f"{row}{col}",
        "grid_x": gx,
        "grid_y": gy,
        "corners_detected": {k: v.tolist() for k, v in corners.items()},
        "tip_px": tip.tolist(),
    }


def draw_result(frame_path, result):
    """Draw overlay showing corners, tip, and prediction."""
    img = cv2.imread(str(frame_path))

    corners = result["corners_detected"]
    pts = []
    for name in KP_NAMES:
        if name in corners:
            x, y = int(corners[name][0]), int(corners[name][1])
            pts.append([x, y])
            cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(img, name, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(pts) >= 3:
        cv2.polylines(img, [np.array(pts, np.int32)], True, (255, 255, 255), 2)

    tx, ty = int(result["tip_px"][0]), int(result["tip_px"][1])
    cv2.circle(img, (tx, ty), 8, (0, 0, 255), -1)
    cv2.putText(img, "tip", (tx + 10, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    well = result["well"]
    gx, gy = result["grid_x"], result["grid_y"]
    cv2.putText(img, f"Pred: {well}  grid=({gx:.2f}, {gy:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Well prediction from FPV frame + tip position")
    parser.add_argument("--frame", required=True, help="Path to FPV frame image")
    parser.add_argument("--tip", nargs=2, type=float, metavar=("X", "Y"),
                        help="Tip position in normalized coords (0-1)")
    parser.add_argument("--tip-px", nargs=2, type=float, metavar=("X", "Y"),
                        help="Tip position in pixel coords")
    parser.add_argument("--model", default=None, help="Path to model weights (default: final.pt)")
    parser.add_argument("--show", action="store_true", help="Show overlay visualization")
    parser.add_argument("--save", default=None, help="Save overlay image to path")
    args = parser.parse_args()

    if args.tip is None and args.tip_px is None:
        print("Error: provide either --tip (normalized) or --tip-px (pixels)")
        sys.exit(1)

    model = load_model(args.model)
    result = predict_well(
        model, args.frame,
        tip_norm=args.tip,
        tip_px=args.tip_px,
    )

    if result is None:
        print("Failed to detect plate corners in the frame.")
        sys.exit(1)

    print(json.dumps({
        "well_row": result["well_row"],
        "well_column": result["well_column"],
        "well": result["well"],
        "grid_x": round(result["grid_x"], 3),
        "grid_y": round(result["grid_y"], 3),
    }, indent=2))

    if args.show or args.save:
        overlay = draw_result(args.frame, result)
        if args.save:
            cv2.imwrite(args.save, overlay)
            print(f"Overlay saved to {args.save}")
        if args.show:
            cv2.imshow("Well Prediction", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
