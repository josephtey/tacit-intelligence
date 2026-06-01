"""Train RF commit-frame detector using 5-fold stratified CV.

Uses the same features and hyperparameters as train_commit_frame_rf.py,
but with clip-level stratified folds from cv_folds.json instead of LOPO.

Usage:
    python experiments/train_rf_cv_folds.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Reuse feature extraction from existing script
from train import (
    _load_frames_gray,
    _load_frames_color,
    _extract_hand_landmarks,
    compute_clip_features,
    FRAMES_DIR,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELLED_FRAMES_PATH = PROJECT_DIR / "pipette_well_dataset" / "labelled_frames.json"
FOLDS_PATH = PROJECT_DIR / "experiments" / "cv_folds.json"
MODELS_DIR = Path(__file__).resolve().parent / "cv_folds"


def main():
    from sklearn.ensemble import RandomForestClassifier

    # Load folds config
    with open(FOLDS_PATH) as f:
        folds_config = json.load(f)
    folds = folds_config["folds"]
    print(f"Loaded {len(folds)}-fold CV config ({folds_config['total_clips']} clips)")

    # Load ground truth
    with open(LABELLED_FRAMES_PATH) as f:
        labelled_frames = json.load(f)

    # Build clip info
    clips = {}
    for clip_name, info in sorted(labelled_frames.items()):
        tv_dir = FRAMES_DIR / f"{clip_name}_Topview"
        if not tv_dir.is_dir():
            print(f"  SKIP {clip_name} — no Topview frames dir")
            continue
        clips[clip_name] = {
            'clip_name': clip_name,
            'plate': clip_name.split("_clip")[0],
            'start_frame': info['start_frame'],
            'end_frame': info['end_frame'],
            'tv_dir': str(tv_dir),
        }

    print(f"Found {len(clips)} clips with frames")

    # Extract features for all clips
    print("\nExtracting features for all clips...")
    t0 = time.time()
    feature_names = None
    for i, (clip_name, clip) in enumerate(sorted(clips.items())):
        print(f"  [{i+1}/{len(clips)}] {clip_name}...", end=" ", flush=True)
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

    # 5-fold CV
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    total_in_window = 0
    total_clips_count = 0

    print(f"\n{'='*60}")
    print("5-Fold Stratified CV")
    print(f"{'='*60}")

    for fold in folds:
        fold_idx = fold["fold"]
        train_ids = set(fold["train_clips"])
        test_ids = set(fold["test_clips"])

        train_clips = [clips[c] for c in train_ids if c in clips]
        test_clips = [clips[c] for c in test_ids if c in clips]

        X_train = np.vstack([c['features'] for c in train_clips])
        y_train = np.concatenate([c['labels'] for c in train_clips])

        print(f"\nFold {fold_idx}: {len(test_clips)} test, {len(train_clips)} train")
        print(f"  Train: {X_train.shape[0]} frames ({int(y_train.sum())} pos)")

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        model_path = MODELS_DIR / f"fold_{fold_idx}.joblib"
        joblib.dump(rf, model_path)
        print(f"  Saved: {model_path}")

        fold_correct = 0
        for clip in test_clips:
            probs = rf.predict_proba(clip['features'])[:, 1]
            probs_smooth = gaussian_filter1d(probs, sigma=3)
            predicted_frame = int(np.argmax(probs_smooth))
            in_window = clip['start_frame'] <= predicted_frame <= clip['end_frame']
            if in_window:
                fold_correct += 1

        pct = 100 * fold_correct / len(test_clips) if test_clips else 0
        print(f"  Result: {fold_correct}/{len(test_clips)} in_window ({pct:.0f}%)")
        total_in_window += fold_correct
        total_clips_count += len(test_clips)

    print(f"\n{'='*60}")
    print(f"Overall: {total_in_window}/{total_clips_count} in_window "
          f"({100*total_in_window/total_clips_count:.1f}%)")
    print(f"{'='*60}")

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    all_clips_list = list(clips.values())
    X_all = np.vstack([c['features'] for c in all_clips_list])
    y_all = np.concatenate([c['labels'] for c in all_clips_list])

    rf_final = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X_all, y_all)
    final_path = MODELS_DIR / "final.joblib"
    joblib.dump(rf_final, final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
