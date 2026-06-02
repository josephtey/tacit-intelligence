# Scientific Knowledge Capture with Vision Models

[![Download the Project Writeup (PDF)](https://img.shields.io/badge/Download-Project_Writeup_(PDF)-1f6feb?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](writeup.pdf)

Predicts which well(s) of a 96-well plate a pipette dispenses into, from a paired FPV + top-view video clip.

## Background

Scientific progress depends on knowing what was actually done — yet most lab work is unobserved. Protocols record what was *intended*, but the real execution (which wells were touched, in what order) survives only as tacit know-how in a researcher's hands, a root cause of the reproducibility crisis. Software made opaque execution legible through observability: logs and traces turned invisible work into a queryable record. The physical sciences have no equivalent — we see a protocol and a result, but never the trace in between.

This project asks whether vision models can recover that missing trace from ordinary video. We treat **"which well did the pipette dispense into"** as the atomic, machine-readable event of a wet-lab experiment. Stacking these events yields an execution log of physical work — the substrate for verification, reproducibility, and the demonstration data that future scientific automation will learn from.

## Problem

Predict the dispensed well(s) of a 96-well plate from a time-synchronized **(FPV, top-down)** clip pair, output as `{well_row, well_column}`.

**Goals** — exact-match well coordinates; generalize to unseen plates; work for all pipette types (single, 8-channel, 12-channel).

**Constraints** — only ~100 clips over 7 plates; output is an 8×12 grid (not a flat label set); < 1 hr of human labeling; near-zero marginal inference cost; < 20 min to run all 100 clips.

## Hypothesis

96-way classification is infeasible with ~100 clips. Instead we reframe the task as **geometry** — corner detection → homography → tip localization in a canonical grid frame → quantize — so each clip teaches transferable plate *structure* rather than a specific label. In this data-scarce regime, geometric inductive biases should generalize to unseen plates better than direct classifiers or zero-shot VLMs.

## Pipeline

Three-stage end-to-end system (`src/e2e_pipeline.py`), each trained on ~100 cheaply-labeled clips:

1. **Commit-frame detection** — Random Forest on motion + MediaPipe hand-skeleton features finds the dispensing frame (top-view).
2. **Plate corner detection** — YOLO keypoint model finds the 4 plate corners (FPV), then warps to a canonical plate view via homography.
3. **Tip localization + type** — HeatNet v2 CNN (heatmap head + type head) localizes the tip on the warped plate and classifies pipette type, then quantizes to a well.

## Results

Evaluated head-to-head against a direct classifier, a zero-shot VLM, and a single-pass keypoint baseline under one matched **leave-one-plate-out (LOPO)** protocol, plus an error-budget chart attributing accuracy to each stage.

Per-stage (LOPO, conditioned on the previous stage succeeding):

| Stage | Accuracy | Notes |
|-------|----------|-------|
| Commit-frame detection | ~80% | hits a valid frame |
| Plate corner detection | ~70% | drops on multi-col/multi-row due to occlusions |
| Tip localization | ~73.3% | |

Corner detection under occlusion is the current bottleneck.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# CLI prediction
python predict.py pipette_well_dataset/Plate_1_clip_0001_FPV.mp4 \
                  pipette_well_dataset/Plate_1_clip_0001_Topview.mp4

# Labeler UI — backend (port 8765) + frontend
python labeler/server.py
cd labeler && npm install && npm run dev
```

`predict.py` outputs `{clip_id_FPV, clip_id_Topview, wells_prediction: [{well_row, well_column}, ...]}`.

## Train & Evaluate

```bash
# Train one model per fold + a final all-data model (folds in experiments/cv_folds.json)
cd models/1_frame_detection && python train_cv.py   # stage 1
cd models/2_plate_detector  && python train_cv.py   # stage 2
cd models/heatnet_v2        && python train_cv.py   # stage 3

# Evaluate (results -> experiments/e2e_runs/<run_name>/results.json)
python3 experiments/run_e2e_cv_eval.py      # held-out CV
python3 experiments/run_e2e_lopo_eval.py    # leave-one-plate-out
python3 experiments/run_e2e_train_eval.py   # train-set upper bound
```

## Project Structure

```
predict.py                 # CLI entry point
src/e2e_pipeline.py        # 3-stage pipeline
models/                    # 1_frame_detection · 2_plate_detector · heatnet_v2 (code + weights)
experiments/               # CV/LOPO/baseline eval scripts + e2e_runs/ results
labeler/                   # visualization UI (E2E Runs + Live Inference)
pipette_well_dataset/      # clip pairs + label JSONs
```
