"""Commit frame detection using Random Forest with LOPO cross-validation.

Combines motion features and MediaPipe hand skeleton features.
Trains one RF per plate fold, saves models and detection.json.

Usage:
    python3 experiments/train_commit_frame_rf.py
"""

import json
import os
import glob
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# ============================================================
# Paths
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELLED_FRAMES_PATH = PROJECT_DIR / "pipette_well_dataset" / "labelled_frames.json"
FRAMES_DIR = PROJECT_DIR / "experiments" / "frames"
MODELS_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "experiments" / "runs"
HAND_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

PLATES = ["Plate_1", "Plate_2", "Plate_3", "Plate_4", "Plate_5", "Plate_9", "Plate_10"]

# ============================================================
# Signal computation (from commit_frame_detector.py)
# ============================================================

def _load_frames_gray(clip_dir):
    paths = sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))
    return [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (320, 240)) for p in paths]


def _load_frames_color(clip_dir):
    paths = sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))
    return [cv2.imread(p) for p in paths]


def _compute_motion(frames):
    motion = np.zeros(len(frames))
    for i in range(1, len(frames)):
        prev = cv2.GaussianBlur(frames[i - 1], (5, 5), 0)
        curr = cv2.GaussianBlur(frames[i], (5, 5), 0)
        motion[i] = np.mean(cv2.absdiff(prev, curr))
    motion[0] = motion[1] if len(frames) > 1 else 0
    return motion


def _smooth(signal, window=15):
    if len(signal) < 7:
        return signal.copy()
    win = min(window, len(signal))
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return signal.copy()
    return savgol_filter(signal, window_length=win, polyorder=3)


def _compute_forward_diff(frames, K=5):
    n = len(frames)
    fwd = np.zeros(n)
    for i in range(n):
        diffs = []
        base = cv2.GaussianBlur(frames[i], (5, 5), 0)
        for j in range(1, K + 1):
            if i + j < n:
                comp = cv2.GaussianBlur(frames[i + j], (5, 5), 0)
                diffs.append(np.mean(cv2.absdiff(base, comp)))
        fwd[i] = np.mean(diffs) if diffs else 0
    return fwd


# ============================================================
# MediaPipe hand landmark extraction
# ============================================================

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.05,
            min_hand_presence_confidence=0.05,
        )
        _landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return _landmarker


def _extract_hand_landmarks(color_frames):
    """Extract hand landmarks for all frames. Returns (n_frames, 21, 3) array.
    If no hand detected, that frame's landmarks are all zeros."""
    import mediapipe as mp

    landmarker = _get_landmarker()
    n = len(color_frames)
    all_landmarks = np.zeros((n, 21, 3))

    for i, frame in enumerate(color_frames):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            # Pick the hand with highest confidence (first one)
            hand = result.hand_landmarks[0]
            for j, lm in enumerate(hand):
                all_landmarks[i, j] = [lm.x, lm.y, lm.z]

    return all_landmarks


# ============================================================
# Feature extraction
# ============================================================

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
# PIP joints for curl angles
THUMB_IP = 3; THUMB_MCP_J = 2
INDEX_PIP = 6; INDEX_DIP = 7
MIDDLE_PIP = 10; MIDDLE_DIP = 11
RING_PIP = 14; RING_DIP = 15
PINKY_PIP = 18; PINKY_DIP = 19

FINGERTIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCP_IDS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PIP_TRIPLETS = [
    (THUMB_MCP_J, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
]

WINDOWS = [5, 10, 15, 20, 30]


def _dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def _angle(a, b, c):
    """Angle at point b formed by segments a-b and b-c, in radians."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1, 1))


def _safe_stat(arr, func, default=0.0):
    if len(arr) == 0:
        return default
    return func(arr)


def compute_clip_features(gray_frames, color_frames, landmarks):
    """Compute per-frame feature matrix for one clip.
    Returns (n_frames, n_features) array and feature names."""
    n = len(gray_frames)

    # --- Motion signals ---
    motion_raw = _compute_motion(gray_frames)
    motion_sm = _smooth(motion_raw)
    fwd_5 = _smooth(_compute_forward_diff(gray_frames, K=5))
    fwd_10 = _smooth(_compute_forward_diff(gray_frames, K=10))

    motion_max = np.max(motion_sm) + 1e-6
    fwd5_max = np.max(fwd_5) + 1e-6
    fwd10_max = np.max(fwd_10) + 1e-6

    velocity = np.gradient(motion_sm)
    accel = np.gradient(velocity)
    fwd5_vel = np.gradient(fwd_5)
    fwd5_accel = np.gradient(fwd5_vel)

    cum_energy = np.cumsum(motion_raw)
    total_energy = cum_energy[-1] + 1e-6
    cum_fwd = np.cumsum(fwd_5)
    total_fwd = cum_fwd[-1] + 1e-6

    motion_sorted = np.sort(motion_sm)

    # --- Skeleton per-frame features ---
    hand_detected = np.zeros(n)
    wrist_x = np.zeros(n)
    wrist_y = np.zeros(n)
    centroid_x = np.zeros(n)
    centroid_y = np.zeros(n)
    fingertip_wrist_dists = np.zeros((n, 5))
    inter_finger_dists = np.zeros((n, 4))
    hand_spread = np.zeros(n)
    thumb_pinky_dist = np.zeros(n)
    finger_curls = np.zeros((n, 5))
    wrist_mcp_dists = np.zeros((n, 4))
    palm_width = np.zeros(n)
    bbox_size = np.zeros(n)

    for i in range(n):
        lm = landmarks[i]  # (21, 3)
        if np.any(lm != 0):
            hand_detected[i] = 1.0
            wrist_x[i] = lm[WRIST, 0]
            wrist_y[i] = lm[WRIST, 1]
            centroid_x[i] = np.mean(lm[:, 0])
            centroid_y[i] = np.mean(lm[:, 1])

            # Fingertip-to-wrist distances
            for fi, tip_id in enumerate(FINGERTIP_IDS):
                fingertip_wrist_dists[i, fi] = _dist(lm[tip_id, :2], lm[WRIST, :2])

            # Inter-finger distances (adjacent pairs)
            for fi in range(4):
                inter_finger_dists[i, fi] = _dist(
                    lm[FINGERTIP_IDS[fi], :2], lm[FINGERTIP_IDS[fi + 1], :2]
                )

            # Hand spread (max fingertip pair distance)
            max_d = 0
            for fi in range(5):
                for fj in range(fi + 1, 5):
                    d = _dist(lm[FINGERTIP_IDS[fi], :2], lm[FINGERTIP_IDS[fj], :2])
                    if d > max_d:
                        max_d = d
            hand_spread[i] = max_d
            thumb_pinky_dist[i] = _dist(lm[THUMB_TIP, :2], lm[PINKY_TIP, :2])

            # Finger curl angles
            for fi, (a, b, c) in enumerate(PIP_TRIPLETS):
                finger_curls[i, fi] = _angle(lm[a, :2], lm[b, :2], lm[c, :2])

            # Wrist-to-MCP distances
            for fi, mcp_id in enumerate(MCP_IDS):
                wrist_mcp_dists[i, fi] = _dist(lm[WRIST, :2], lm[mcp_id, :2])

            # Palm width
            palm_width[i] = _dist(lm[INDEX_MCP, :2], lm[PINKY_MCP, :2])

            # Bbox size
            xs = lm[:, 0]
            ys = lm[:, 1]
            bbox_size[i] = (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))

    # Wrist velocity for temporal features
    wrist_vel = np.sqrt(np.gradient(wrist_x) ** 2 + np.gradient(wrist_y) ** 2)

    # --- Build feature matrix ---
    features = []
    names = []

    for i in range(n):
        f = []

        # A. Motion features
        # Base signals (4)
        f.extend([motion_raw[i], motion_sm[i], fwd_5[i], fwd_10[i]])
        # Normalized (4)
        f.extend([motion_raw[i] / motion_max, motion_sm[i] / motion_max,
                  fwd_5[i] / fwd5_max, fwd_10[i] / fwd10_max])
        # Position (2)
        f.extend([i / n, i])
        # Velocity/accel (4)
        f.extend([velocity[i], accel[i], fwd5_vel[i], fwd5_accel[i]])

        # Backward window stats (4 stats × 5 windows = 20)
        for W in WINDOWS:
            seg = motion_sm[max(0, i - W):i] if i > 0 else np.array([motion_sm[i]])
            f.extend([
                _safe_stat(seg, np.mean),
                _safe_stat(seg, np.max),
                _safe_stat(seg, np.std),
                _safe_stat(seg, np.min),
            ])

        # Forward window stats (4 × 5 = 20)
        for W in WINDOWS:
            seg = motion_sm[i:min(n, i + W)]
            f.extend([
                _safe_stat(seg, np.mean),
                _safe_stat(seg, np.max),
                _safe_stat(seg, np.std),
                _safe_stat(seg, np.min),
            ])

        # Contrast features (4 × 5 = 20)
        for W in WINDOWS:
            bseg = motion_sm[max(0, i - W):i] if i > 0 else np.array([motion_sm[i]])
            fseg = motion_sm[i:min(n, i + W)]
            pre_max = _safe_stat(bseg, np.max, motion_sm[i])
            post_max = _safe_stat(fseg, np.max, motion_sm[i])
            pre_contrast = pre_max - motion_sm[i]
            post_contrast = post_max - motion_sm[i]
            total_contrast = pre_contrast + post_contrast
            contrast_ratio = (pre_contrast + 1e-6) / (post_contrast + 1e-6)
            f.extend([pre_contrast, post_contrast, total_contrast, contrast_ratio])

        # Cumulative energy (4)
        f.extend([
            cum_energy[i] / total_energy,
            1 - cum_energy[i] / total_energy,
            cum_fwd[i] / total_fwd,
            1 - cum_fwd[i] / total_fwd,
        ])

        # Percentile ranks (3)
        motion_rank = np.searchsorted(motion_sorted, motion_sm[i]) / n
        fwd5_rank = np.searchsorted(np.sort(fwd_5), fwd_5[i]) / n
        f.extend([motion_rank, fwd5_rank, motion_rank + fwd5_rank])

        # Quality-inspired (4)
        stillness = 1.0 - (motion_sm[i] - np.min(motion_sm)) / (np.max(motion_sm) - np.min(motion_sm) + 1e-6)
        stability = 1.0 - fwd_5[i] / fwd5_max
        rel_pos = i / n
        edge_penalty = 0.1 if (rel_pos < 0.03 or rel_pos > 0.97) else (0.5 if (rel_pos < 0.05 or rel_pos > 0.95) else 1.0)
        fwd_stability = 1.0 - fwd_10[i] / fwd10_max
        f.extend([stillness, stability, edge_penalty, fwd_stability])

        # B. Skeleton features
        # Hand detected (1)
        f.append(hand_detected[i])
        # Wrist position (2)
        f.extend([wrist_x[i], wrist_y[i]])
        # Fingertip-to-wrist distances (5)
        f.extend(fingertip_wrist_dists[i].tolist())
        # Inter-finger distances (4)
        f.extend(inter_finger_dists[i].tolist())
        # Hand spread, thumb-pinky (2)
        f.extend([hand_spread[i], thumb_pinky_dist[i]])
        # Finger curl angles (5)
        f.extend(finger_curls[i].tolist())
        # Wrist-to-MCP distances (4)
        f.extend(wrist_mcp_dists[i].tolist())
        # Palm width (1)
        f.append(palm_width[i])
        # Centroid (2)
        f.extend([centroid_x[i], centroid_y[i]])
        # Bbox size (1)
        f.append(bbox_size[i])

        # Temporal diffs for skeleton (W=1,3,5,10 × 6 signals = 24)
        for W in [1, 3, 5, 10]:
            prev_i = max(0, i - W)
            f.extend([
                wrist_x[i] - wrist_x[prev_i],
                wrist_y[i] - wrist_y[prev_i],
                centroid_x[i] - centroid_x[prev_i],
                centroid_y[i] - centroid_y[prev_i],
                hand_spread[i] - hand_spread[prev_i],
                bbox_size[i] - bbox_size[prev_i],
            ])

        # Temporal window stats for wrist velocity (2 × 5 = 10)
        for W in WINDOWS:
            seg = wrist_vel[max(0, i - W):i + 1]
            f.extend([
                _safe_stat(seg, np.mean),
                _safe_stat(seg, np.std),
            ])

        features.append(f)

    # Build feature names (only on first call, for reporting)
    if not hasattr(compute_clip_features, '_names'):
        names = []
        names.extend(['motion_raw', 'motion_sm', 'fwd_5', 'fwd_10'])
        names.extend(['motion_raw_norm', 'motion_sm_norm', 'fwd_5_norm', 'fwd_10_norm'])
        names.extend(['rel_pos', 'abs_frame'])
        names.extend(['velocity', 'accel', 'fwd5_vel', 'fwd5_accel'])
        for W in WINDOWS:
            for stat in ['mean', 'max', 'std', 'min']:
                names.append(f'bwd_{stat}_W{W}')
        for W in WINDOWS:
            for stat in ['mean', 'max', 'std', 'min']:
                names.append(f'fwd_{stat}_W{W}')
        for W in WINDOWS:
            for stat in ['pre_contrast', 'post_contrast', 'total_contrast', 'contrast_ratio']:
                names.append(f'{stat}_W{W}')
        names.extend(['cum_energy', 'rem_energy', 'cum_fwd_energy', 'rem_fwd_energy'])
        names.extend(['motion_rank', 'fwd5_rank', 'combined_rank'])
        names.extend(['stillness', 'stability', 'edge_penalty', 'fwd_stability'])
        names.append('hand_detected')
        names.extend(['wrist_x', 'wrist_y'])
        names.extend([f'tip_wrist_dist_{i}' for i in range(5)])
        names.extend([f'inter_finger_dist_{i}' for i in range(4)])
        names.extend(['hand_spread', 'thumb_pinky_dist'])
        names.extend([f'finger_curl_{i}' for i in range(5)])
        names.extend([f'wrist_mcp_dist_{i}' for i in range(4)])
        names.append('palm_width')
        names.extend(['centroid_x', 'centroid_y'])
        names.append('bbox_size')
        for W in [1, 3, 5, 10]:
            for sig in ['wrist_x', 'wrist_y', 'centroid_x', 'centroid_y', 'spread', 'bbox']:
                names.append(f'delta_{sig}_W{W}')
        for W in WINDOWS:
            names.extend([f'wrist_vel_mean_W{W}', f'wrist_vel_std_W{W}'])
        compute_clip_features._names = names

    return np.array(features), getattr(compute_clip_features, '_names', None)


# ============================================================
# Main
# ============================================================

def main():
    from sklearn.ensemble import RandomForestClassifier

    # Load ground truth
    with open(LABELLED_FRAMES_PATH) as f:
        labelled_frames = json.load(f)

    # Build clip info
    clips = []
    for clip_name, info in sorted(labelled_frames.items()):
        plate = clip_name.split("_clip")[0]
        tv_dir = FRAMES_DIR / f"{clip_name}_Topview"
        if not tv_dir.is_dir():
            print(f"  SKIP {clip_name} — no Topview frames dir")
            continue
        clips.append({
            'clip_name': clip_name,
            'plate': plate,
            'start_frame': info['start_frame'],
            'end_frame': info['end_frame'],
            'tv_dir': str(tv_dir),
        })

    print(f"Loaded {len(clips)} clips across {len(set(c['plate'] for c in clips))} plates")

    # Extract features for all clips (cache to avoid recomputation)
    print("\nExtracting features for all clips...")
    t0 = time.time()
    feature_names = None
    for i, clip in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] {clip['clip_name']}...", end=" ", flush=True)
        gray_frames = _load_frames_gray(clip['tv_dir'])
        color_frames = _load_frames_color(clip['tv_dir'])
        landmarks = _extract_hand_landmarks(color_frames)
        features, names = compute_clip_features(gray_frames, color_frames, landmarks)
        if feature_names is None:
            feature_names = names
        n = len(gray_frames)
        labels = np.zeros(n)
        labels[clip['start_frame']:clip['end_frame'] + 1] = 1.0
        clip['features'] = features
        clip['labels'] = labels
        clip['n_frames'] = n
        print(f"({n} frames, {int(labels.sum())} pos)")
    print(f"Feature extraction done in {time.time() - t0:.1f}s")
    print(f"Feature dimensions: {clips[0]['features'].shape[1]} features per frame")

    # LOPO cross-validation
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_commit_frame_rf_lopo"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_in_window = 0
    total_clips = 0
    per_plate_stats = {}

    print(f"\n{'='*60}")
    print("LOPO Cross-Validation")
    print(f"{'='*60}")

    for fold_plate in PLATES:
        train_clips = [c for c in clips if c['plate'] != fold_plate]
        test_clips = [c for c in clips if c['plate'] == fold_plate]

        if not test_clips:
            continue

        # Build training data
        X_train = np.vstack([c['features'] for c in train_clips])
        y_train = np.concatenate([c['labels'] for c in train_clips])

        print(f"\nFold: hold out {fold_plate} ({len(test_clips)} clips)")
        print(f"  Train: {X_train.shape[0]} frames ({int(y_train.sum())} pos, {int((1-y_train).sum())} neg)")

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Save model
        model_path = MODELS_DIR / f"fold_{fold_plate}.joblib"
        joblib.dump(rf, model_path)
        print(f"  Model saved: {model_path}")

        # Predict on test clips
        fold_correct = 0
        fold_total = len(test_clips)

        for clip in test_clips:
            probs = rf.predict_proba(clip['features'])[:, 1]
            probs_smooth = gaussian_filter1d(probs, sigma=3)
            predicted_frame = int(np.argmax(probs_smooth))

            in_window = clip['start_frame'] <= predicted_frame <= clip['end_frame']
            if in_window:
                fold_correct += 1

            all_results.append({
                'clip_name': f"{clip['clip_name']}_Topview",
                'plate': clip['plate'],
                'predicted_frame': predicted_frame,
                'gt_frame': None,
                'diff': None,
                'n_frames': clip['n_frames'],
                'start_frame': clip['start_frame'],
                'end_frame': clip['end_frame'],
                'in_window': in_window,
                'probabilities': [round(float(p), 4) for p in probs_smooth],
            })

        total_in_window += fold_correct
        total_clips += fold_total
        per_plate_stats[fold_plate] = f"{fold_correct}/{fold_total}"
        print(f"  Result: {fold_correct}/{fold_total} in_window ({fold_correct/fold_total*100:.0f}%)")

    # Overall results
    print(f"\n{'='*60}")
    print(f"Overall: {total_in_window}/{total_clips} in_window ({total_in_window/total_clips*100:.1f}%)")
    print(f"{'='*60}")
    print("\nPer-plate breakdown:")
    for plate in PLATES:
        if plate in per_plate_stats:
            print(f"  {plate:12s}: {per_plate_stats[plate]}")

    # Feature importances (average across folds)
    print("\nTop 20 feature importances (averaged across folds):")
    avg_importances = np.zeros(len(feature_names))
    n_folds = 0
    for fold_plate in PLATES:
        model_path = MODELS_DIR / f"fold_{fold_plate}.joblib"
        if model_path.exists():
            rf = joblib.load(model_path)
            avg_importances += rf.feature_importances_
            n_folds += 1
    if n_folds > 0:
        avg_importances /= n_folds
        top_idx = np.argsort(avg_importances)[::-1][:20]
        for rank, idx in enumerate(top_idx):
            print(f"  {rank+1:2d}. {feature_names[idx]:30s} {avg_importances[idx]:.4f}")

    # Train final model on ALL data (no holdout)
    print(f"\n{'='*60}")
    print("Training final model on all 100 clips (no holdout)...")
    X_all = np.vstack([c['features'] for c in clips])
    y_all = np.concatenate([c['labels'] for c in clips])
    print(f"  All data: {X_all.shape[0]} frames ({int(y_all.sum())} pos, {int((1-y_all).sum())} neg)")

    rf_final = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X_all, y_all)
    final_model_path = MODELS_DIR / "final_all_data.joblib"
    joblib.dump(rf_final, final_model_path)
    print(f"  Final model saved: {final_model_path}")
    print(f"{'='*60}")

    # Save detection.json
    output = {
        "detector": "commit_frame_rf_lopo",
        "timestamp": timestamp,
        "config": {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "n_features": len(feature_names),
            "feature_names": feature_names,
        },
        "summary": {
            "n_clips": total_clips,
            "in_window": total_in_window,
            "in_window_pct": round(100 * total_in_window / total_clips, 1),
            "per_plate": per_plate_stats,
        },
        "clips": all_results,
    }

    out_path = run_dir / "detection.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
