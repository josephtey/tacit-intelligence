"""Train HeatNet v2: multi-tip heatmaps + pipette type classification.

Builds on HeatNet v1 with:
- Multi-tip heatmap targets (element-wise max of Gaussian blobs)
- Type classification head (single/multi-col/multi-row)
- Joint loss: 1.0 * MSE_heatmap + 0.1 * CE_type (inverse-frequency weighted)
- LOPO eval with predicted type AND oracle type accuracy

Usage:
    python experiments/train_heatnet_v2.py              # LOPO eval + final model
    python experiments/train_heatnet_v2.py --overlays   # Also generate overlay images
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
with open(DATA_DIR / "labels.json") as f:
    data = json.load(f)
clips = data["clips"]
PADDING = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# H100-optimized settings
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 150
LR = 1e-3
HMAP_SIGMA = 5
TYPE_LOSS_WEIGHT = 0.1
NUM_TYPES = 3  # 0=single, 1=multi-col, 2=multi-row
TYPE_NAMES = ["single", "multi-col", "multi-row"]


def pixel_to_well(x, y, res=640):
    if res == 640:
        gx = (x - PADDING) / 40.0
        gy = (y - PADDING) / 40.0
    else:
        gx = (x - PADDING) / (1080.0 / 11)
        gy = (y - PADDING) / (760.0 / 7)
    col = int(np.clip(round(gx), 0, 11))
    row = int(np.clip(round(gy), 0, 7))
    return row, col


def grid_to_well_name(row, col):
    return f"{'ABCDEFGH'[row]}{col + 1}"


def is_correct_prediction(pred_row, pred_col, clip, pipette_type=None):
    """Type-aware correctness check. Uses pipette_type override if provided."""
    pt = pipette_type if pipette_type is not None else clip["pipette_type"]
    if pt == "single":
        return pred_row == clip["gt_well_row"] and pred_col == clip["gt_well_col"]
    elif pt == "multi-col":
        return pred_row == clip["gt_well_row"]  # only row matters
    elif pt == "multi-row":
        return pred_col == clip["gt_well_col"]  # only column matters
    return False


# ===== Dataset =====

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

        # Get ALL tip coordinates
        tips_key = f"tips_{self.res}"
        tips = [(t["x"], t["y"]) for t in c[tips_key]]

        type_label = c["type_label"]

        # Resize to 320x240
        scale_x = 320 / img.shape[1]
        scale_y = 240 / img.shape[0]
        img = cv2.resize(img, (320, 240))
        tips = [(tx * scale_x, ty * scale_y) for tx, ty in tips]

        if self.augment and aug_type > 0:
            if aug_type == 1:  # horizontal flip
                img = img[:, ::-1, :].copy()
                tips = [(320 - tx, ty) for tx, ty in tips]
            elif aug_type == 2:  # brightness jitter
                delta = np.random.uniform(-20, 20)
                img = np.clip(img + delta, 0, 255)
            elif aug_type == 3:  # contrast jitter
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255)
            elif aug_type == 4:  # Gaussian noise
                noise = np.random.normal(0, 8, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0, 255)
            elif aug_type == 5:  # small shift
                dx = np.random.randint(-10, 10)
                dy = np.random.randint(-10, 10)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (320, 240))
                tips = [(tx + dx, ty + dy) for tx, ty in tips]
            elif aug_type == 6:  # blur
                k = np.random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif aug_type == 7:  # channel shuffle
                perm = np.random.permutation(3)
                img = img[:, :, perm]

        # Normalize
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1).astype(np.float32)

        # Build multi-tip heatmap at 1/4 resolution: 80x60
        # Use element-wise max to keep targets in [0, 1]
        hmap = np.zeros((60, 80), dtype=np.float32)
        yy, xx = np.mgrid[0:60, 0:80]
        for tx, ty in tips:
            hx, hy = tx / 4, ty / 4
            blob = np.exp(-((xx - hx)**2 + (yy - hy)**2) / (2 * HMAP_SIGMA**2))
            hmap = np.maximum(hmap, blob)

        return (torch.FloatTensor(img),
                torch.FloatTensor(hmap[None]),
                torch.LongTensor([type_label]))


# ===== Model =====

class HeatNetV2(nn.Module):
    """Heatmap CNN with pipette type classification head."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 160x120
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 80x60
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # Heatmap head (unchanged from v1)
        self.hmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        # Type classification head
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


# ===== Training and Evaluation =====

def compute_class_weights(train_indices):
    """Compute inverse-frequency class weights for CE loss."""
    counts = np.zeros(NUM_TYPES, dtype=np.float32)
    for i in train_indices:
        counts[clips[i]["type_label"]] += 1
    # Inverse frequency, normalized so weights sum to NUM_TYPES
    weights = np.zeros(NUM_TYPES, dtype=np.float32)
    for t in range(NUM_TYPES):
        if counts[t] > 0:
            weights[t] = len(train_indices) / (NUM_TYPES * counts[t])
        else:
            weights[t] = 0.0
    return torch.FloatTensor(weights)


def predict_from_heatmap(model, img_path, device):
    """Run inference on one image, return (pred_row, pred_col, tip_x, tip_y, hmap, pred_type)."""
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_resized = cv2.resize(img_rgb, (320, 240))
    img_norm = img_resized / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_t = torch.FloatTensor(img_norm.transpose(2, 0, 1))[None].to(device)

    with torch.no_grad():
        hmap, type_logits = model(img_t)
        hmap = hmap[0, 0].cpu().numpy()  # 60x80
        pred_type = type_logits.argmax(dim=1).item()

    peak_y, peak_x = np.unravel_index(hmap.argmax(), hmap.shape)
    tip_x = peak_x * 4 * (640 / 320)
    tip_y = peak_y * 4 * (480 / 240)
    pred_row, pred_col = pixel_to_well(tip_x, tip_y, 640)
    return pred_row, pred_col, tip_x, tip_y, hmap, pred_type


def train_model(train_indices, epochs=EPOCHS, lr=LR):
    """Train HeatNetV2 on given indices, return trained model."""
    train_ds = WellDataset(train_indices, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = HeatNetV2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Inverse-frequency class weights
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
            print(f"  Epoch {epoch}: hmap_loss={epoch_hmap_loss/n:.6f}, type_loss={epoch_type_loss/n:.4f}")

    return model


def run_lopo(generate_overlays=False):
    """Leave-one-plate-out evaluation with type classification + well prediction."""
    print(f"Device: {DEVICE}")
    print(f"Dataset: {len(clips)} clips")
    print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, Epochs: {EPOCHS}")
    print(f"Type loss weight: {TYPE_LOSS_WEIGHT}")

    plates = sorted(set(c["plate"] for c in clips))
    plate_to_idx = defaultdict(list)
    for i, c in enumerate(clips):
        plate_to_idx[c["plate"]].append(i)

    overlay_dir = PROJECT_DIR / "experiments" / "heatnet_v2_results" / "overlays"
    if generate_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    total_correct_pred = 0  # using predicted type
    total_correct_oracle = 0  # using GT type
    total_count = 0
    type_pred_correct = 0  # type classification accuracy
    type_confusion = np.zeros((NUM_TYPES, NUM_TYPES), dtype=int)  # [gt][pred]
    plate_correct_pred = defaultdict(int)
    plate_correct_oracle = defaultdict(int)
    plate_total = defaultdict(int)
    ptype_correct_pred = defaultdict(int)
    ptype_correct_oracle = defaultdict(int)
    ptype_total = defaultdict(int)

    for test_plate in plates:
        test_idx = plate_to_idx[test_plate]
        train_idx = [i for p in plates if p != test_plate for i in plate_to_idx[p]]

        test_types = set(clips[i]["pipette_type"] for i in test_idx)
        train_types = set(clips[i]["pipette_type"] for i in train_idx)
        print(f"\n--- Test {test_plate} ({len(test_idx)} clips, {len(train_idx)} train) ---")
        print(f"  Test types: {test_types}, Train types: {train_types}")

        model = train_model(train_idx)
        model.eval()

        # Save per-fold model
        fold_dir = Path(__file__).resolve().parent / "folds"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / f"heatnet_v2_exclude_{test_plate}.pt")

        if generate_overlays:
            plate_dir = overlay_dir / test_plate
            plate_dir.mkdir(exist_ok=True)

        fold_correct_pred = 0
        fold_correct_oracle = 0
        fold_type_correct = 0

        for idx in test_idx:
            c = clips[idx]
            img_path = DATA_DIR / "images_640" / f"{c['clip_id']}.jpg"
            pred_row, pred_col, tip_x, tip_y, hmap, pred_type_idx = predict_from_heatmap(model, img_path, DEVICE)

            gt_type_idx = c["type_label"]
            pred_type_name = TYPE_NAMES[pred_type_idx]
            gt_type_name = c["pipette_type"]

            # Type accuracy
            type_correct_hit = (pred_type_idx == gt_type_idx)
            if type_correct_hit:
                type_pred_correct += 1
                fold_type_correct += 1
            type_confusion[gt_type_idx][pred_type_idx] += 1

            # Well accuracy with PREDICTED type
            hit_pred = is_correct_prediction(pred_row, pred_col, c, pred_type_name)
            if hit_pred:
                fold_correct_pred += 1

            # Well accuracy with ORACLE (GT) type
            hit_oracle = is_correct_prediction(pred_row, pred_col, c)
            if hit_oracle:
                fold_correct_oracle += 1

            pt = c["pipette_type"]
            ptype_correct_pred[pt] += int(hit_pred)
            ptype_correct_oracle[pt] += int(hit_oracle)
            ptype_total[pt] += 1

            if generate_overlays:
                raw_img = cv2.imread(str(img_path))
                gold_tip = (c["tip_640"]["x"], c["tip_640"]["y"])
                overlay = draw_overlay(
                    raw_img, hmap, (tip_x, tip_y), gold_tip,
                    pred_row, pred_col, c, hit_oracle,
                    pred_type_name=pred_type_name)
                status = "correct" if hit_oracle else "wrong"
                cv2.imwrite(str(plate_dir / f"{status}_{c['clip_id']}.jpg"), overlay)

        pct_pred = 100 * fold_correct_pred / len(test_idx) if test_idx else 0
        pct_oracle = 100 * fold_correct_oracle / len(test_idx) if test_idx else 0
        pct_type = 100 * fold_type_correct / len(test_idx) if test_idx else 0
        print(f"  Type acc: {fold_type_correct}/{len(test_idx)} = {pct_type:.1f}%")
        print(f"  Well acc (pred type): {fold_correct_pred}/{len(test_idx)} = {pct_pred:.1f}%")
        print(f"  Well acc (oracle):    {fold_correct_oracle}/{len(test_idx)} = {pct_oracle:.1f}%")

        total_correct_pred += fold_correct_pred
        total_correct_oracle += fold_correct_oracle
        total_count += len(test_idx)
        plate_correct_pred[test_plate] = fold_correct_pred
        plate_correct_oracle[test_plate] = fold_correct_oracle
        plate_total[test_plate] = len(test_idx)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"LOPO OVERALL:")
    print(f"  Type classification: {type_pred_correct}/{total_count} = {100*type_pred_correct/total_count:.1f}%")
    print(f"  Well acc (pred type): {total_correct_pred}/{total_count} = {100*total_correct_pred/total_count:.1f}%")
    print(f"  Well acc (oracle):    {total_correct_oracle}/{total_count} = {100*total_correct_oracle/total_count:.1f}%")

    print(f"\nType confusion matrix (rows=GT, cols=pred):")
    print(f"  {'':15s} {'single':>8s} {'multi-col':>10s} {'multi-row':>10s}")
    for gt_idx, gt_name in enumerate(TYPE_NAMES):
        row = type_confusion[gt_idx]
        print(f"  {gt_name:15s} {row[0]:8d} {row[1]:10d} {row[2]:10d}")

    print(f"\nBy plate:")
    for p in plates:
        pp, po, pt = plate_correct_pred[p], plate_correct_oracle[p], plate_total[p]
        types = set(clips[i]["pipette_type"] for i in plate_to_idx[p])
        print(f"  {p}: pred={pp}/{pt}={100*pp/pt:.1f}%  oracle={po}/{pt}={100*po/pt:.1f}%  ({types})")

    print(f"\nBy pipette type:")
    for pt in TYPE_NAMES:
        if ptype_total[pt] > 0:
            pp = ptype_correct_pred[pt]
            po = ptype_correct_oracle[pt]
            t = ptype_total[pt]
            print(f"  {pt}: pred={pp}/{t}={100*pp/t:.1f}%  oracle={po}/{t}={100*po/t:.1f}%")
    print(f"{'=' * 70}")

    if generate_overlays:
        nc = sum(1 for f in overlay_dir.rglob("correct_*.jpg"))
        nw = sum(1 for f in overlay_dir.rglob("wrong_*.jpg"))
        print(f"\nOverlays saved to {overlay_dir}/ ({nc} correct, {nw} wrong)")

    return total_correct_pred, total_correct_oracle, total_count


def train_and_save_final():
    """Train final model on ALL data and save it."""
    print(f"\n{'=' * 70}")
    print(f"Training FINAL model on all {len(clips)} clips...")
    print(f"{'=' * 70}")

    all_indices = list(range(len(clips)))
    model = train_model(all_indices)

    save_dir = Path(__file__).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "heatnet_v2.pt")

    metadata = {
        "architecture": "HeatNetV2",
        "input_size": [3, 240, 320],
        "heatmap_output_size": [1, 60, 80],
        "type_classes": TYPE_NAMES,
        "heatmap_sigma": HMAP_SIGMA,
        "type_loss_weight": TYPE_LOSS_WEIGHT,
        "training_clips": len(clips),
        "pipette_types": TYPE_NAMES,
        "plates": sorted(set(c["plate"] for c in clips)),
        "warp_config": {"width": 640, "height": 480, "padding": 100},
        "training_params": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "augmentation": "8x (flip, brightness, contrast, noise, shift, blur, channel_shuffle)",
        },
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {save_dir / 'heatnet_v2.pt'}")
    print(f"Metadata saved to {save_dir / 'metadata.json'}")

    # Quick validation on training data
    model.eval()
    correct = 0
    type_correct = 0
    for i, c in enumerate(clips):
        img_path = DATA_DIR / "images_640" / f"{c['clip_id']}.jpg"
        pred_row, pred_col, _, _, _, pred_type_idx = predict_from_heatmap(model, img_path, DEVICE)
        if is_correct_prediction(pred_row, pred_col, c):
            correct += 1
        if pred_type_idx == c["type_label"]:
            type_correct += 1
    print(f"Training set well accuracy: {correct}/{len(clips)} = {100*correct/len(clips):.1f}%")
    print(f"Training set type accuracy: {type_correct}/{len(clips)} = {100*type_correct/len(clips):.1f}%")


def draw_overlay(warped_img, hmap_60x80, pred_tip, gold_tip, pred_row, pred_col,
                 clip, hit, pred_type_name=None):
    """Draw overlay with heatmap activation, tips, and well grid."""
    overlay = warped_img.copy()
    pt = clip["pipette_type"]

    # Resize heatmap and blend
    hmap_norm = (hmap_60x80 - hmap_60x80.min()) / (hmap_60x80.max() - hmap_60x80.min() + 1e-8)
    hmap_full = cv2.resize(hmap_norm, (640, 480))
    hmap_color = cv2.applyColorMap((hmap_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = hmap_full > 0.15
    for ch in range(3):
        overlay[:, :, ch] = np.where(
            mask,
            (overlay[:, :, ch] * 0.5 + hmap_color[:, :, ch] * 0.5).astype(np.uint8),
            overlay[:, :, ch])

    # Grid lines
    for col in range(12):
        x = int(PADDING + col * (640 - 2 * PADDING) / 11.0)
        cv2.line(overlay, (x, PADDING), (x, 480 - PADDING), (50, 50, 50), 1)
    for row in range(8):
        y = int(PADDING + row * (480 - 2 * PADDING) / 7.0)
        cv2.line(overlay, (PADDING, y), (640 - PADDING, y), (50, 50, 50), 1)
    cv2.rectangle(overlay, (PADDING, PADDING), (640 - PADDING, 480 - PADDING), (100, 100, 100), 2)

    # Highlight ground truth region for multi-well
    cell_h = (480 - 2 * PADDING) / 7.0
    cell_w = (640 - 2 * PADDING) / 11.0
    if pt == "multi-col" and clip["gt_well_row"] >= 0:
        gt_row = clip["gt_well_row"]
        y_center = PADDING + gt_row * cell_h
        y_top = max(PADDING, int(y_center - cell_h / 2))
        y_bot = min(480 - PADDING, int(y_center + cell_h / 2))
        if y_bot > y_top:
            band = overlay[y_top:y_bot, PADDING:640-PADDING].copy()
            green_band = np.zeros_like(band)
            green_band[:, :, 1] = 100
            overlay[y_top:y_bot, PADDING:640-PADDING] = cv2.addWeighted(band, 0.7, green_band, 0.3, 0)
    elif pt == "multi-row" and clip["gt_well_col"] >= 0:
        gt_col = clip["gt_well_col"]
        x_center = PADDING + gt_col * cell_w
        x_left = max(PADDING, int(x_center - cell_w / 2))
        x_right = min(640 - PADDING, int(x_center + cell_w / 2))
        if x_right > x_left:
            band = overlay[PADDING:480-PADDING, x_left:x_right].copy()
            green_band = np.zeros_like(band)
            green_band[:, :, 1] = 100
            overlay[PADDING:480-PADDING, x_left:x_right] = cv2.addWeighted(band, 0.7, green_band, 0.3, 0)

    # Draw all gold tips (green circles, smaller for multi)
    tips_640 = clip.get("tips_640", [])
    if len(tips_640) > 1:
        for t in tips_640:
            tx, ty = int(t["x"]), int(t["y"])
            cv2.circle(overlay, (tx, ty), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (tx, ty), 4, (0, 0, 0), 1, cv2.LINE_AA)
    elif gold_tip is not None:
        gx, gy = int(gold_tip[0]), int(gold_tip[1])
        cv2.circle(overlay, (gx, gy), 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (gx, gy), 8, (0, 0, 0), 2, cv2.LINE_AA)

    # Predicted tip (purple circle)
    if pred_tip is not None:
        px, py = int(pred_tip[0]), int(pred_tip[1])
        cv2.circle(overlay, (px, py), 8, (94, 63, 244), -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), 8, (0, 0, 0), 2, cv2.LINE_AA)

    # Status text
    if pt == "multi-col":
        gt_str = f"Row {chr(clip['gt_well_row'] + ord('A'))} (multi-col)"
        pred_str = f"Row {chr(pred_row + ord('A'))}"
    elif pt == "multi-row":
        gt_str = f"Col {clip['gt_well_col'] + 1} (multi-row)"
        pred_str = f"Col {pred_col + 1}"
    else:
        gt_str = grid_to_well_name(clip["gt_well_row"], clip["gt_well_col"])
        pred_str = grid_to_well_name(pred_row, pred_col)

    status = "CORRECT" if hit else "WRONG"
    color = (0, 255, 0) if hit else (0, 0, 255)
    type_str = f" [pred:{pred_type_name}]" if pred_type_name else ""
    cv2.putText(overlay, f"Pred: {pred_str}  GT: {gt_str}  [{status}]{type_str}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    return overlay


if __name__ == "__main__":
    generate_overlays = "--overlays" in sys.argv

    print("=" * 70)
    print("HeatNet V2: Multi-Tip Heatmaps + Type Classification")
    print(f"Clips: {len(clips)} | Plates: {sorted(set(c['plate'] for c in clips))}")
    type_counts = defaultdict(int)
    for c in clips:
        type_counts[c["pipette_type"]] += 1
    print(f"Types: {dict(type_counts)}")
    print("=" * 70)

    # LOPO evaluation
    run_lopo(generate_overlays=generate_overlays)

    # Train and save final model on all data
    train_and_save_final()
