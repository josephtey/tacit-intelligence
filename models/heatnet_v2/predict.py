"""HeatNet v2 inference script.

Takes a warped plate image (640x480 with 100px padding) and optionally a
pipette type override, runs HeatNetV2 to predict the tip location and pipette
type, and outputs wells_prediction in the challenge JSON format.

Usage:
    # Auto-detect pipette type from model
    python models/heatnet_v2/predict.py --image path/to/warped.jpg

    # Override pipette type (skip classification)
    python models/heatnet_v2/predict.py --image path/to/warped.jpg --type single

    # From raw frame + corner annotations (auto-warps)
    python models/heatnet_v2/predict.py --frame path/to/frame.jpg \
        --corners '{"A1": [0.66, 0.55], "A12": [0.87, 0.50], "H1": [0.68, 0.79], "H12": [0.91, 0.74]}'

    # Use a specific fold model
    python models/heatnet_v2/predict.py --image path/to/warped.jpg \
        --model models/heatnet_v2/folds/heatnet_v2_exclude_Plate_10.pt

Output (JSON):
    {
        "pred_row": 0,
        "pred_col": 5,
        "predicted_type": "single",
        "wells_prediction": [{"well_row": "A", "well_column": "6"}]
    }
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ===== Model definition (must match training) =====

NUM_TYPES = 3
TYPE_NAMES = ["single", "multi-col", "multi-row"]


class HeatNetV2(nn.Module):
    """Heatmap CNN with pipette type classification head."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.hmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_TYPES),
        )

    def forward(self, x):
        features = self.enc(x)
        heatmap = self.hmap_head(features)
        type_logits = self.type_head(features)
        return heatmap, type_logits


# ===== Constants =====

PADDING = 100
WARP_W, WARP_H = 640, 480
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = MODEL_DIR / "heatnet_v2.pt"


# ===== Warping utilities =====

def warp_frame(frame, corners, w=WARP_W, h=WARP_H, pad=PADDING):
    """Warp a raw frame to canonical plate view using 4 corner annotations.

    Args:
        frame: BGR image (numpy array)
        corners: dict with keys "A1", "A12", "H1", "H12", each a [x_norm, y_norm] pair
    Returns:
        warped: 640x480 BGR image with plate in canonical position
    """
    img_h, img_w = frame.shape[:2]
    dst = {
        "A1": np.array([pad, pad], dtype=np.float32),
        "A12": np.array([w - pad, pad], dtype=np.float32),
        "H12": np.array([w - pad, h - pad], dtype=np.float32),
        "H1": np.array([pad, h - pad], dtype=np.float32),
    }
    src_pts, dst_pts = [], []
    for name in ["A1", "A12", "H12", "H1"]:
        pt = corners[name]
        src_pts.append([pt[0] * img_w, pt[1] * img_h])
        dst_pts.append(dst[name])
    M = cv2.getPerspectiveTransform(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
    )
    return cv2.warpPerspective(frame, M, (w, h))


# ===== Core prediction =====

def load_model(model_path=None, device="cpu"):
    """Load HeatNetV2 from a .pt state dict file."""
    path = Path(model_path) if model_path else DEFAULT_MODEL
    model = HeatNetV2()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_tip(model, warped_img, device="cpu"):
    """Predict tip location and type from a 640x480 warped plate image.

    Returns:
        pred_row: int 0-7 (A-H)
        pred_col: int 0-11 (1-12)
        tip_x: float, predicted tip x in 640x480 coords
        tip_y: float, predicted tip y in 640x480 coords
        heatmap: numpy array (60x80), raw heatmap output
        pred_type: str, predicted pipette type
    """
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_resized = cv2.resize(img_rgb, (320, 240))
    img_norm = img_resized / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_t = torch.FloatTensor(img_norm.transpose(2, 0, 1))[None].to(device)

    with torch.no_grad():
        hmap, type_logits = model(img_t)
        hmap = hmap[0, 0].cpu().numpy()
        pred_type_idx = type_logits.argmax(dim=1).item()

    peak_y, peak_x = np.unravel_index(hmap.argmax(), hmap.shape)
    tip_x = peak_x * 4 * (WARP_W / 320)
    tip_y = peak_y * 4 * (WARP_H / 240)

    gx = (tip_x - PADDING) / 40.0
    gy = (tip_y - PADDING) / 40.0
    pred_col = int(np.clip(round(gx), 0, 11))
    pred_row = int(np.clip(round(gy), 0, 7))

    return pred_row, pred_col, tip_x, tip_y, hmap, TYPE_NAMES[pred_type_idx]


def expand_prediction(pred_row, pred_col, pipette_type):
    """Expand a single (row, col) prediction to the full wells_prediction list."""
    if pipette_type == "single":
        return [{"well_row": chr(pred_row + ord("A")),
                 "well_column": str(pred_col + 1)}]
    elif pipette_type == "multi-col":
        row = chr(pred_row + ord("A"))
        return [{"well_row": row, "well_column": str(c)} for c in range(1, 13)]
    elif pipette_type == "multi-row":
        col = str(pred_col + 1)
        return [{"well_row": chr(r + ord("A")), "well_column": col} for r in range(8)]
    else:
        raise ValueError(f"Unknown pipette_type: {pipette_type}")


def predict(warped_img, pipette_type=None, model=None, device="cpu"):
    """Full prediction pipeline: image -> wells_prediction JSON.

    Args:
        warped_img: 640x480 BGR warped plate image
        pipette_type: optional override ("single", "multi-col", "multi-row").
                      If None, model predicts the type.
        model: loaded HeatNetV2 model (if None, loads default)
        device: "cpu" or "cuda"

    Returns:
        dict with pred_row, pred_col, predicted_type, tip_x, tip_y, wells_prediction
    """
    if model is None:
        model = load_model(device=device)
    pred_row, pred_col, tip_x, tip_y, _, model_type = predict_tip(model, warped_img, device)

    # Use override type if provided, otherwise use model prediction
    used_type = pipette_type if pipette_type is not None else model_type
    wells = expand_prediction(pred_row, pred_col, used_type)

    return {
        "pred_row": pred_row,
        "pred_col": pred_col,
        "predicted_type": model_type,
        "tip_x": round(tip_x, 2),
        "tip_y": round(tip_y, 2),
        "wells_prediction": wells,
    }


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(description="HeatNet v2 well prediction")
    parser.add_argument("--image", help="Path to warped 640x480 plate image")
    parser.add_argument("--frame", help="Path to raw frame (requires --corners)")
    parser.add_argument("--corners", help="JSON string of corner annotations (for --frame)")
    parser.add_argument("--type", default=None, choices=["single", "multi-col", "multi-row"],
                        help="Pipette type override (default: model predicts)")
    parser.add_argument("--model", default=None, help="Path to .pt model (default: heatnet_v2.pt)")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    args = parser.parse_args()

    if not args.image and not args.frame:
        parser.error("Provide either --image (warped) or --frame (raw + --corners)")
    if args.frame and not args.corners:
        parser.error("--frame requires --corners")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    if args.image:
        warped = cv2.imread(args.image)
        if warped is None:
            print(f"Error: cannot read {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"Error: cannot read {args.frame}", file=sys.stderr)
            sys.exit(1)
        corners = json.loads(args.corners)
        warped = warp_frame(frame, corners)

    result = predict(warped, args.type, model, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
