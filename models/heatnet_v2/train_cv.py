"""Train HeatNet v2 using 5-fold stratified CV.

Uses the same architecture and training params as train_heatnet_v2.py,
but with clip-level stratified folds from cv_folds.json instead of LOPO.

Usage:
    python experiments/train_heatnet_v2_cv_folds.py
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import sys
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "experiments" / "tip_detection_dataset"
FOLDS_PATH = PROJECT_DIR / "experiments" / "cv_folds.json"
MODELS_DIR = Path(__file__).resolve().parent / "cv_folds"

with open(DATA_DIR / "labels.json") as f:
    data = json.load(f)
clips = data["clips"]

PADDING = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 150
LR = 1e-3
HMAP_SIGMA = 5
TYPE_LOSS_WEIGHT = 0.1
NUM_TYPES = 3
TYPE_NAMES = ["single", "multi-col", "multi-row"]

# Build clip_id -> index mapping
CLIP_ID_TO_IDX = {c["clip_id"]: i for i, c in enumerate(clips)}


def pixel_to_well(x, y):
    gx = (x - PADDING) / 40.0
    gy = (y - PADDING) / 40.0
    col = int(np.clip(round(gx), 0, 11))
    row = int(np.clip(round(gy), 0, 7))
    return row, col


def is_correct_prediction(pred_row, pred_col, clip, pipette_type=None):
    pt = pipette_type if pipette_type is not None else clip["pipette_type"]
    if pt == "single":
        return pred_row == clip["gt_well_row"] and pred_col == clip["gt_well_col"]
    elif pt == "multi-col":
        return pred_row == clip["gt_well_row"]
    elif pt == "multi-row":
        return pred_col == clip["gt_well_col"]
    return False


class WellDataset(Dataset):
    def __init__(self, indices, augment=False, res=640):
        self.indices = indices
        self.augment = augment
        self.res = res
        self.aug_factor = 8 if augment else 1

    def __len__(self):
        return len(self.indices) * self.aug_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.indices)
        aug_type = idx // len(self.indices)

        c = clips[self.indices[real_idx]]
        img = cv2.imread(str(DATA_DIR / f"images_{self.res}" / f"{c['clip_id']}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        tips_key = f"tips_{self.res}"
        tips = [(t["x"], t["y"]) for t in c[tips_key]]
        type_label = c["type_label"]

        scale_x = 320 / img.shape[1]
        scale_y = 240 / img.shape[0]
        img = cv2.resize(img, (320, 240))
        tips = [(tx * scale_x, ty * scale_y) for tx, ty in tips]

        if self.augment and aug_type > 0:
            if aug_type == 1:
                img = img[:, ::-1, :].copy()
                tips = [(320 - tx, ty) for tx, ty in tips]
            elif aug_type == 2:
                delta = np.random.uniform(-20, 20)
                img = np.clip(img + delta, 0, 255)
            elif aug_type == 3:
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255)
            elif aug_type == 4:
                noise = np.random.normal(0, 8, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0, 255)
            elif aug_type == 5:
                dx = np.random.randint(-10, 10)
                dy = np.random.randint(-10, 10)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (320, 240))
                tips = [(tx + dx, ty + dy) for tx, ty in tips]
            elif aug_type == 6:
                k = np.random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif aug_type == 7:
                perm = np.random.permutation(3)
                img = img[:, :, perm]

        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1).astype(np.float32)

        hmap = np.zeros((60, 80), dtype=np.float32)
        yy, xx = np.mgrid[0:60, 0:80]
        for tx, ty in tips:
            hx, hy = tx / 4, ty / 4
            blob = np.exp(-((xx - hx)**2 + (yy - hy)**2) / (2 * HMAP_SIGMA**2))
            hmap = np.maximum(hmap, blob)

        return (torch.FloatTensor(img),
                torch.FloatTensor(hmap[None]),
                torch.LongTensor([type_label]))


class HeatNetV2(nn.Module):
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


def compute_class_weights(train_indices):
    counts = np.zeros(NUM_TYPES, dtype=np.float32)
    for i in train_indices:
        counts[clips[i]["type_label"]] += 1
    weights = np.zeros(NUM_TYPES, dtype=np.float32)
    for t in range(NUM_TYPES):
        if counts[t] > 0:
            weights[t] = len(train_indices) / (NUM_TYPES * counts[t])
    return torch.FloatTensor(weights)


def predict_from_heatmap(model, img_path, device):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_resized = cv2.resize(img_rgb, (320, 240))
    img_norm = img_resized / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_t = torch.FloatTensor(img_norm.transpose(2, 0, 1))[None].to(device)

    with torch.no_grad():
        hmap, type_logits = model(img_t)
        hmap = hmap[0, 0].cpu().numpy()
        pred_type = type_logits.argmax(dim=1).item()

    peak_y, peak_x = np.unravel_index(hmap.argmax(), hmap.shape)
    tip_x = peak_x * 4 * (640 / 320)
    tip_y = peak_y * 4 * (480 / 240)
    pred_row, pred_col = pixel_to_well(tip_x, tip_y)
    return pred_row, pred_col, pred_type


def train_model(train_indices, epochs=EPOCHS, lr=LR):
    train_ds = WellDataset(train_indices, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = HeatNetV2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    class_weights = compute_class_weights(train_indices).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        epoch_hmap_loss = 0
        epoch_type_loss = 0
        for imgs, hmap_targets, type_labels in train_loader:
            imgs = imgs.to(DEVICE)
            hmap_targets = hmap_targets.to(DEVICE)
            type_labels = type_labels.to(DEVICE).squeeze(1)

            hmap_preds, type_logits = model(imgs)
            loss_hmap = F.mse_loss(hmap_preds, hmap_targets)
            loss_type = ce_loss_fn(type_logits, type_labels)
            loss = loss_hmap + TYPE_LOSS_WEIGHT * loss_type

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_hmap_loss += loss_hmap.item()
            epoch_type_loss += loss_type.item()
        scheduler.step()

        if epoch % 30 == 0:
            n = len(train_loader)
            print(f"  Epoch {epoch}: hmap={epoch_hmap_loss/n:.6f}, type={epoch_type_loss/n:.4f}")

    return model


def main():
    with open(FOLDS_PATH) as f:
        folds_config = json.load(f)
    folds = folds_config["folds"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"HeatNet v2: 5-Fold Stratified CV")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {len(clips)} clips (tip detection subset)")
    print(f"Epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    print(f"{'='*60}")

    total_correct_pred = 0
    total_correct_oracle = 0
    total_type_correct = 0
    total_count = 0
    type_confusion = np.zeros((NUM_TYPES, NUM_TYPES), dtype=int)

    for fold in folds:
        fold_idx = fold["fold"]
        train_ids = set(fold["train_clips"])
        test_ids = set(fold["test_clips"])

        # Map to indices in the 86-clip dataset
        train_indices = [CLIP_ID_TO_IDX[c] for c in train_ids if c in CLIP_ID_TO_IDX]
        test_indices = [CLIP_ID_TO_IDX[c] for c in test_ids if c in CLIP_ID_TO_IDX]

        missing_train = len(train_ids) - len(train_indices)
        missing_test = len(test_ids) - len(test_indices)

        print(f"\n--- Fold {fold_idx}: {len(test_indices)} test, {len(train_indices)} train ---")
        if missing_train or missing_test:
            print(f"  (skipped {missing_train} train, {missing_test} test clips not in tip dataset)")

        model = train_model(train_indices)
        model.eval()

        torch.save(model.state_dict(), MODELS_DIR / f"fold_{fold_idx}.pt")

        fold_correct_pred = 0
        fold_correct_oracle = 0
        fold_type_correct = 0

        for idx in test_indices:
            c = clips[idx]
            img_path = DATA_DIR / "images_640" / f"{c['clip_id']}.jpg"
            pred_row, pred_col, pred_type_idx = predict_from_heatmap(model, img_path, DEVICE)

            gt_type_idx = c["type_label"]
            pred_type_name = TYPE_NAMES[pred_type_idx]

            if pred_type_idx == gt_type_idx:
                fold_type_correct += 1
            type_confusion[gt_type_idx][pred_type_idx] += 1

            if is_correct_prediction(pred_row, pred_col, c, pred_type_name):
                fold_correct_pred += 1
            if is_correct_prediction(pred_row, pred_col, c):
                fold_correct_oracle += 1

        n = len(test_indices)
        print(f"  Type acc:  {fold_type_correct}/{n} = {100*fold_type_correct/n:.1f}%")
        print(f"  Well pred: {fold_correct_pred}/{n} = {100*fold_correct_pred/n:.1f}%")
        print(f"  Well orac: {fold_correct_oracle}/{n} = {100*fold_correct_oracle/n:.1f}%")

        total_correct_pred += fold_correct_pred
        total_correct_oracle += fold_correct_oracle
        total_type_correct += fold_type_correct
        total_count += n

    print(f"\n{'='*60}")
    print(f"OVERALL ({total_count} clips evaluated):")
    print(f"  Type:        {total_type_correct}/{total_count} = {100*total_type_correct/total_count:.1f}%")
    print(f"  Well (pred): {total_correct_pred}/{total_count} = {100*total_correct_pred/total_count:.1f}%")
    print(f"  Well (orac): {total_correct_oracle}/{total_count} = {100*total_correct_oracle/total_count:.1f}%")
    print(f"\nType confusion (rows=GT, cols=pred):")
    print(f"  {'':15s} {'single':>8s} {'multi-col':>10s} {'multi-row':>10s}")
    for gt_idx, gt_name in enumerate(TYPE_NAMES):
        row = type_confusion[gt_idx]
        print(f"  {gt_name:15s} {row[0]:8d} {row[1]:10d} {row[2]:10d}")
    print(f"{'='*60}")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
