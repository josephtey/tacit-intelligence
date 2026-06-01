"""End-to-end well prediction pipeline.

Wires together 3 stages:
  1. RF commit frame detection (from Topview video)
  2. YOLO plate corner detection (from FPV commit frame)
  3. HeatNet tip detection (from warped plate image)

Supports fold-specific models (for LOPO eval) and final models (for inference).
Optionally generates overlay images at each stage.

Usage:
    from src.e2e_pipeline import EndToEndPipeline

    # Final models (inference)
    pipe = EndToEndPipeline()
    result = pipe.predict("clip_FPV.mp4", "clip_Topview.mp4")

    # Fold-specific models (LOPO eval)
    pipe = EndToEndPipeline(fold_plate="Plate_1", generate_overlays=True)
    result = pipe.predict("clip_FPV.mp4", "clip_Topview.mp4",
                          ground_truth=[{"well_row": "A", "well_column": "1"}],
                          overlay_dir="output/")
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
HAND_MODEL_PATH = MODELS_DIR / "1_frame_detection" / "hand_landmarker.task"

# Import RF feature extraction from training script
sys.path.insert(0, str(MODELS_DIR / "1_frame_detection"))
from train import (
    compute_clip_features,
    _compute_motion,
    _smooth,
    _extract_hand_landmarks,
)

# Import plate detector
sys.path.insert(0, str(MODELS_DIR / "2_plate_detector"))
from inference import (
    load_model as load_yolo_model,
    extract_corners,
    compute_homography,
    KP_NAMES,
    GRID_COORDS,
)

# Import tip detector (HeatNet v2 with pipette type classification)
sys.path.insert(0, str(MODELS_DIR / "heatnet_v2"))
from predict import (
    HeatNetV2,
    load_model as load_heatnet_model,
    predict_tip,
    warp_frame,
    expand_prediction,
    PADDING,
    WARP_W,
    WARP_H,
)

# Fold mapping: sorted plate names → YOLO fold indices
PLATES_SORTED = ["Plate_1", "Plate_10", "Plate_2", "Plate_3",
                 "Plate_4", "Plate_5", "Plate_9"]
YOLO_FOLD_MAP = {plate: i for i, plate in enumerate(PLATES_SORTED)}

# Corner colors for overlays
CORNER_COLORS = {
    "A1": (128, 222, 74),    # green
    "A12": (250, 165, 96),   # orange
    "H12": (22, 115, 249),   # blue
    "H1": (252, 132, 192),   # pink
}


class EndToEndPipeline:
    """Three-stage pipeline: RF commit frame -> YOLO corners -> HeatNet tip."""

    def __init__(
        self,
        fold_plate=None,
        cv_fold=None,
        rf_model_path=None,
        yolo_model_path=None,
        heatnet_model_path=None,
        device="cpu",
        generate_overlays=False,
    ):
        self.device = device
        self.generate_overlays = generate_overlays

        # Resolve model paths
        if cv_fold is not None:
            # Stratified CV fold models
            self.rf_path = rf_model_path or (
                MODELS_DIR / "1_frame_detection" / "cv_folds" / f"fold_{cv_fold}.joblib"
            )
            self.yolo_path = yolo_model_path or (
                MODELS_DIR / "2_plate_detector" / "cv_folds" / f"fold_{cv_fold}.pt"
            )
            self.heatnet_path = heatnet_model_path or (
                MODELS_DIR / "heatnet_v2" / "cv_folds" / f"fold_{cv_fold}.pt"
            )
        elif fold_plate:
            # LOPO fold models
            self.rf_path = rf_model_path or (
                MODELS_DIR / "1_frame_detection" / f"fold_{fold_plate}.joblib"
            )
            yolo_idx = YOLO_FOLD_MAP[fold_plate]
            self.yolo_path = yolo_model_path or (
                MODELS_DIR / "2_plate_detector" / f"fold_{yolo_idx}_{fold_plate}.pt"
            )
            self.heatnet_path = heatnet_model_path or (
                MODELS_DIR / "heatnet_v2" / "folds" / f"heatnet_v2_exclude_{fold_plate}.pt"
            )
        else:
            # Final models (inference)
            self.rf_path = rf_model_path or (
                MODELS_DIR / "1_frame_detection" / "final_all_data.joblib"
            )
            self.yolo_path = yolo_model_path or (
                MODELS_DIR / "2_plate_detector" / "final.pt"
            )
            self.heatnet_path = heatnet_model_path or (
                MODELS_DIR / "heatnet_v2" / "heatnet_v2.pt"
            )

        # Load models
        self.rf_model = joblib.load(str(self.rf_path))
        self.yolo_model = load_yolo_model(str(self.yolo_path))
        self.heatnet_model = load_heatnet_model(str(self.heatnet_path), device=device)

    def predict(self, fpv_path, topview_path, ground_truth=None, overlay_dir=None,
                pipette_type=None):
        """Run full 3-stage pipeline.

        Args:
            fpv_path: Path to FPV video
            topview_path: Path to Topview video
            ground_truth: Optional list of {"well_row", "well_column"} dicts
            overlay_dir: Directory to save overlay images (if generate_overlays=True)
            pipette_type: "single", "multi-col", or "multi-row". If None, defaults to "single".

        Returns:
            dict with wells_prediction, stages info, status
        """
        clip_id_fpv = Path(fpv_path).stem
        clip_id_tv = Path(topview_path).stem
        clip_name = clip_id_fpv.replace("_FPV", "")

        result = {
            "clip_id_FPV": clip_id_fpv,
            "clip_id_Topview": clip_id_tv,
            "wells_prediction": [],
            "stages": {},
            "pipette_type": None,
            "status": "ok",
            "timing": {},
            "cost": {"input_tokens": 0, "output_tokens": 0,
                     "thinking_tokens": 0, "usd": 0.0},
        }
        _t_total = time.perf_counter()

        # ── Stage 1: Commit frame detection ──
        _t = time.perf_counter()
        s1 = self._run_stage1(topview_path)
        result["timing"]["stage1_commit"] = round(time.perf_counter() - _t, 3)
        result["stages"]["commit_frame"] = {
            "frame_idx": s1.get("commit_idx"),
            "total_frames": s1.get("total_frames"),
            "status": s1["status"],
        }
        if s1["status"] != "ok":
            result["status"] = "stage1_fail"
            result["wells_prediction"] = [{"well_row": "D", "well_column": "6"}]
            result["timing"]["total"] = round(time.perf_counter() - _t_total, 3)
            if self.generate_overlays and overlay_dir:
                self._generate_overlays(
                    clip_name, overlay_dir, s1, None, None,
                    None, None, ground_truth
                )
            return result

        commit_idx = s1["commit_idx"]

        # ── Extract FPV commit frame ──
        fpv_frames = self._extract_frames_color(fpv_path)
        if not fpv_frames:
            result["status"] = "stage1_fail"
            result["wells_prediction"] = [{"well_row": "D", "well_column": "6"}]
            result["timing"]["total"] = round(time.perf_counter() - _t_total, 3)
            return result

        # Use same frame index from topview, clamped to FPV length
        fpv_idx = min(commit_idx, len(fpv_frames) - 1)
        fpv_frame = fpv_frames[fpv_idx]

        # ── Stage 2: Plate corner detection ──
        _t = time.perf_counter()
        s2 = self._run_stage2(fpv_frame)
        result["timing"]["stage2_corners"] = round(time.perf_counter() - _t, 3)
        result["stages"]["plate_detection"] = {
            "corners": {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in s2.get("corners_px", {}).items()},
            "n_corners": s2.get("n_corners", 0),
            "status": s2["status"],
        }
        if s2["status"] != "ok":
            result["status"] = "stage2_fail"
            result["wells_prediction"] = [{"well_row": "D", "well_column": "6"}]
            result["timing"]["total"] = round(time.perf_counter() - _t_total, 3)
            if self.generate_overlays and overlay_dir:
                self._generate_overlays(
                    clip_name, overlay_dir, s1, s2, None,
                    fpv_frame, None, ground_truth
                )
            return result

        warped_frame = s2["warped_frame"]

        # ── Stage 3: Tip detection ──
        _t = time.perf_counter()
        s3 = self._run_stage3(warped_frame)
        result["timing"]["stage3_tip"] = round(time.perf_counter() - _t, 3)
        result["stages"]["tip_detection"] = {
            "pred_row": s3["pred_row"],
            "pred_col": s3["pred_col"],
            "tip_x": round(s3["tip_x"], 2),
            "tip_y": round(s3["tip_y"], 2),
            "pred_type": s3.get("pred_type", "single"),
            "status": s3["status"],
        }
        if s3["status"] != "ok":
            result["status"] = "stage3_fail"
            result["wells_prediction"] = [{"well_row": "D", "well_column": "6"}]
            result["pipette_type"] = pipette_type or "single"
        else:
            # Use explicit pipette_type if given, else model's prediction
            used_type = pipette_type if pipette_type is not None else s3.get("pred_type", "single")
            wells = expand_prediction(s3["pred_row"], s3["pred_col"], used_type)
            result["pipette_type"] = used_type
            result["wells_prediction"] = wells

        # ── Generate overlays ──
        if self.generate_overlays and overlay_dir:
            self._generate_overlays(
                clip_name, overlay_dir, s1, s2, s3,
                fpv_frame, result["pipette_type"], ground_truth
            )

        result["timing"]["total"] = round(time.perf_counter() - _t_total, 3)
        return result

    # ─── Stage 1: RF Commit Frame ─────────────────────────────

    def _run_stage1(self, topview_path):
        """Extract frames, compute RF features, predict commit frame."""
        try:
            # Extract all frames
            gray_frames = []
            color_frames = []
            cap = cv2.VideoCapture(topview_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                color_frames.append(frame)
                gray = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240)
                )
                gray_frames.append(gray)
            cap.release()

            n = len(gray_frames)
            if n < 3:
                return {"status": "fail", "commit_idx": 0, "total_frames": n}

            # Extract hand landmarks
            landmarks = _extract_hand_landmarks(color_frames)

            # Compute features
            features, _ = compute_clip_features(gray_frames, color_frames, landmarks)

            # Run RF
            probs = self.rf_model.predict_proba(features)[:, 1]
            probs_smooth = gaussian_filter1d(probs, sigma=3)

            # Compute motion signal
            motion = _compute_motion(gray_frames)
            motion_sm = _smooth(motion)

            commit_idx = int(np.argmax(probs_smooth))

            return {
                "status": "ok",
                "commit_idx": commit_idx,
                "total_frames": n,
                "probabilities": probs_smooth,
                "motion_signal": motion_sm,
                "topview_frame": color_frames[commit_idx],
            }
        except Exception as e:
            return {"status": "fail", "error": str(e),
                    "commit_idx": 0, "total_frames": 0}

    # ─── Stage 2: YOLO Plate Detection ────────────────────────

    def _run_stage2(self, fpv_frame):
        """Run YOLO on FPV frame, detect corners, warp."""
        try:
            # Save frame to temp file for YOLO
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, fpv_frame)
                tmp_path = tmp.name

            corners_px = extract_corners(self.yolo_model, tmp_path)
            os.unlink(tmp_path)

            if corners_px is None or len(corners_px) < 3:
                return {"status": "fail", "corners_px": {}, "n_corners": 0}

            n_corners = len(corners_px)

            # Estimate missing corner if only 3 detected
            if n_corners == 3:
                missing = [k for k in KP_NAMES if k not in corners_px][0]
                if missing == "A1":
                    corners_px["A1"] = corners_px["A12"] + corners_px["H1"] - corners_px["H12"]
                elif missing == "A12":
                    corners_px["A12"] = corners_px["A1"] + corners_px["H12"] - corners_px["H1"]
                elif missing == "H12":
                    corners_px["H12"] = corners_px["A12"] + corners_px["H1"] - corners_px["A1"]
                elif missing == "H1":
                    corners_px["H1"] = corners_px["A1"] + corners_px["H12"] - corners_px["A12"]

            # Convert to normalized coords for warp_frame
            h, w = fpv_frame.shape[:2]
            corners_norm = {}
            for name, px in corners_px.items():
                corners_norm[name] = [float(px[0]) / w, float(px[1]) / h]

            # Warp frame to canonical plate view
            warped = warp_frame(fpv_frame, corners_norm)

            return {
                "status": "ok",
                "corners_px": corners_px,
                "corners_norm": corners_norm,
                "n_corners": n_corners,
                "warped_frame": warped,
            }
        except Exception as e:
            return {"status": "fail", "corners_px": {}, "n_corners": 0,
                    "error": str(e)}

    # ─── Stage 3: HeatNet Tip Detection ──────────────────────

    def _run_stage3(self, warped_frame):
        """Run HeatNet v2 on warped image."""
        try:
            pred_row, pred_col, tip_x, tip_y, heatmap, pred_type = predict_tip(
                self.heatnet_model, warped_frame, device=self.device
            )
            return {
                "status": "ok",
                "pred_row": int(pred_row),
                "pred_col": int(pred_col),
                "pred_type": pred_type,
                "tip_x": float(tip_x),
                "tip_y": float(tip_y),
                "heatmap": heatmap,
            }
        except Exception as e:
            return {"status": "fail", "pred_row": 3, "pred_col": 5,
                    "tip_x": 0, "tip_y": 0, "error": str(e)}

    # ─── Helpers ─────────────────────────────────────────────

    def _extract_frames_color(self, video_path):
        """Extract all color frames from a video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    # ─── Overlay Generation ──────────────────────────────────

    def _generate_overlays(self, clip_name, overlay_dir, s1, s2, s3,
                           fpv_frame, pipette_type, ground_truth):
        """Generate and save overlay images for each stage."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(overlay_dir, exist_ok=True)

        # ── Stage 1: Commit frame overlay ──
        self._gen_s1_overlay(clip_name, overlay_dir, s1)

        # ── Stage 2: Corners overlay ──
        if s2 and fpv_frame is not None:
            self._gen_s2_overlay(clip_name, overlay_dir, s2, fpv_frame)

        # ── Stage 3: Warped + heatmap overlay ──
        if s3 and s2 and s2.get("warped_frame") is not None:
            self._gen_s3_overlay(
                clip_name, overlay_dir, s3, s2["warped_frame"],
                pipette_type, ground_truth
            )

        # ── Summary panel ──
        self._gen_summary_overlay(
            clip_name, overlay_dir, s1, s2, s3,
            fpv_frame, pipette_type, ground_truth
        )

    def _gen_s1_overlay(self, clip_name, overlay_dir, s1):
        """Stage 1: Just the commit frame with a label."""
        if s1.get("topview_frame") is not None:
            img = s1["topview_frame"].copy()
            ci = s1.get("commit_idx", "?")
            n = s1.get("total_frames", "?")
            cv2.putText(img, f"Commit Frame {ci}/{n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No frame available", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        path = os.path.join(overlay_dir, f"{clip_name}_s1_commit.jpg")
        cv2.imwrite(path, img)

    def _gen_s2_overlay(self, clip_name, overlay_dir, s2, fpv_frame):
        """Stage 2: FPV frame with corner keypoints drawn."""
        img = fpv_frame.copy()
        corners_px = s2.get("corners_px", {})

        # Draw polygon connecting corners
        pts = []
        for name in KP_NAMES:
            if name in corners_px:
                px = corners_px[name]
                x, y = int(px[0]), int(px[1])
                pts.append([x, y])

        if len(pts) >= 3:
            cv2.polylines(img, [np.array(pts, np.int32)], True, (255, 255, 255), 2)

        # Draw corner keypoints
        for name in KP_NAMES:
            if name in corners_px:
                px = corners_px[name]
                x, y = int(px[0]), int(px[1])
                color = CORNER_COLORS.get(name, (0, 255, 0))
                cv2.circle(img, (x, y), 10, color, -1)
                cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(img, name, (x + 14, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status text
        n = s2.get("n_corners", 0)
        status_text = f"Corners: {n}/4 detected"
        cv2.putText(img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        path = os.path.join(overlay_dir, f"{clip_name}_s2_corners.jpg")
        cv2.imwrite(path, img)

    def _gen_s3_overlay(self, clip_name, overlay_dir, s3, warped_frame,
                        pipette_type, ground_truth):
        """Stage 3: Warped plate with grid + heatmap + prediction + GT."""
        img = warped_frame.copy()

        # Draw well grid
        self._draw_well_grid(img)

        # Overlay heatmap
        if s3.get("heatmap") is not None:
            hmap = s3["heatmap"]
            # Normalize to 0-255
            hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
            hmap_resized = cv2.resize(hmap_norm, (WARP_W, WARP_H))
            hmap_color = cv2.applyColorMap(
                (hmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            # Blend only where heatmap is significant
            mask = hmap_resized > 0.1
            alpha = 0.4
            for c in range(3):
                img[:, :, c] = np.where(
                    mask,
                    (1 - alpha) * img[:, :, c] + alpha * hmap_color[:, :, c],
                    img[:, :, c]
                ).astype(np.uint8)

        # Draw predicted well
        pred_row = s3.get("pred_row", 0)
        pred_col = s3.get("pred_col", 0)
        pred_cx = PADDING + pred_col * 40 + 20
        pred_cy = PADDING + pred_row * 40 + 20
        cv2.circle(img, (pred_cx, pred_cy), 16, (0, 0, 255), 3)
        cv2.circle(img, (pred_cx, pred_cy), 4, (0, 0, 255), -1)

        # Draw tip position
        tip_x = s3.get("tip_x", pred_cx)
        tip_y = s3.get("tip_y", pred_cy)
        cv2.drawMarker(img, (int(tip_x), int(tip_y)), (0, 0, 255),
                        cv2.MARKER_CROSS, 20, 2)

        # Draw ground truth
        if ground_truth:
            for gt in ground_truth:
                gt_row = ord(gt["well_row"]) - ord("A")
                gt_col = int(gt["well_column"]) - 1
                gt_cx = PADDING + gt_col * 40 + 20
                gt_cy = PADDING + gt_row * 40 + 20
                cv2.circle(img, (gt_cx, gt_cy), 14, (0, 255, 0), 3)

        # Labels
        pred_well = f"{chr(pred_row + ord('A'))}{pred_col + 1}"
        label = f"Pred: {pred_well}"
        if pipette_type:
            label += f" ({pipette_type})"
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if ground_truth:
            gt_str = ", ".join(f"{g['well_row']}{g['well_column']}" for g in ground_truth[:3])
            cv2.putText(img, f"GT: {gt_str}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        path = os.path.join(overlay_dir, f"{clip_name}_s3_warped.jpg")
        cv2.imwrite(path, img)

    def _gen_summary_overlay(self, clip_name, overlay_dir, s1, s2, s3,
                             fpv_frame, pipette_type, ground_truth):
        """Combined summary panel with all stages side-by-side."""
        panel_h = 360
        panel_w = 480
        gap = 10
        n_panels = 3
        total_w = n_panels * panel_w + (n_panels - 1) * gap
        header_h = 50
        canvas = np.zeros((header_h + panel_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Header
        status = "OK" if (s3 and s3.get("status") == "ok") else "FAIL"
        pred_str = "?"
        if s3 and s3.get("status") == "ok":
            pred_str = f"{chr(s3['pred_row'] + ord('A'))}{s3['pred_col'] + 1}"
            if pipette_type and pipette_type != "single":
                pred_str += f" ({pipette_type})"

        gt_str = ""
        if ground_truth:
            gt_str = ", ".join(
                f"{g['well_row']}{g['well_column']}" for g in ground_truth[:3]
            )

        correct = False
        if ground_truth and s3 and s3.get("status") == "ok":
            pred_wells = expand_prediction(s3["pred_row"], s3["pred_col"],
                                           pipette_type or "single")
            pred_set = {(w["well_row"], str(w["well_column"])) for w in pred_wells}
            gt_set = {(g["well_row"], str(g["well_column"])) for g in ground_truth}
            correct = pred_set == gt_set

        header_text = f"{clip_name}  |  Pred: {pred_str}"
        if gt_str:
            header_text += f"  |  GT: {gt_str}"
            verdict = "CORRECT" if correct else "WRONG"
            header_text += f"  |  {verdict}"

        header_color = (0, 255, 0) if correct else (0, 0, 255)
        if not ground_truth:
            header_color = (255, 255, 0)
        cv2.putText(canvas, header_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)

        # Panel 1: Topview commit frame
        if s1 and s1.get("topview_frame") is not None:
            p1 = cv2.resize(s1["topview_frame"], (panel_w, panel_h))
            ci = s1.get("commit_idx", "?")
            cv2.putText(p1, f"S1: Commit f{ci}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            canvas[header_h:, 0:panel_w] = p1

        # Panel 2: FPV with corners
        if fpv_frame is not None:
            p2 = cv2.resize(fpv_frame, (panel_w, panel_h))
            if s2 and s2.get("corners_px"):
                h_orig, w_orig = fpv_frame.shape[:2]
                sx = panel_w / w_orig
                sy = panel_h / h_orig
                for name in KP_NAMES:
                    if name in s2["corners_px"]:
                        px = s2["corners_px"][name]
                        x = int(px[0] * sx)
                        y = int(px[1] * sy)
                        color = CORNER_COLORS.get(name, (0, 255, 0))
                        cv2.circle(p2, (x, y), 6, color, -1)
                n_c = s2.get("n_corners", 0)
                cv2.putText(p2, f"S2: {n_c}/4 corners", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            canvas[header_h:, panel_w + gap:2 * panel_w + gap] = p2

        # Panel 3: Warped plate with prediction
        if s2 and s2.get("warped_frame") is not None:
            p3 = cv2.resize(s2["warped_frame"], (panel_w, panel_h))
            if s3 and s3.get("status") == "ok":
                # Scale prediction to panel size
                sx = panel_w / WARP_W
                sy = panel_h / WARP_H
                pred_x = int((PADDING + s3["pred_col"] * 40 + 20) * sx)
                pred_y = int((PADDING + s3["pred_row"] * 40 + 20) * sy)
                cv2.circle(p3, (pred_x, pred_y), 10, (0, 0, 255), 2)

                # Ground truth
                if ground_truth:
                    for gt in ground_truth[:1]:
                        gt_row = ord(gt["well_row"]) - ord("A")
                        gt_col = int(gt["well_column"]) - 1
                        gt_x = int((PADDING + gt_col * 40 + 20) * sx)
                        gt_y = int((PADDING + gt_row * 40 + 20) * sy)
                        cv2.circle(p3, (gt_x, gt_y), 10, (0, 255, 0), 2)

                cv2.putText(p3, f"S3: {pred_str}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            canvas[header_h:, 2 * (panel_w + gap):] = p3

        path = os.path.join(overlay_dir, f"{clip_name}_summary.jpg")
        cv2.imwrite(path, canvas)

    def _draw_well_grid(self, img):
        """Draw 96-well grid overlay on a warped 640x480 image."""
        # Grid lines
        for col in range(13):
            x = PADDING + col * 40
            cv2.line(img, (x, PADDING), (x, WARP_H - PADDING),
                     (200, 200, 200), 1)
        for row in range(9):
            y = PADDING + row * 40
            cv2.line(img, (PADDING, y), (WARP_W - PADDING, y),
                     (200, 200, 200), 1)

        # Row labels (A-H)
        for row in range(8):
            label = chr(ord("A") + row)
            y = PADDING + row * 40 + 25
            cv2.putText(img, label, (PADDING - 25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Column labels (1-12)
        for col in range(12):
            label = str(col + 1)
            x = PADDING + col * 40 + 12
            cv2.putText(img, label, (x, PADDING - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
