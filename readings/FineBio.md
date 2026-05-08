# FineBio: A Fine-Grained Video Dataset of Biological Experiments with Hierarchical Annotation

**Authors:** Takuma Yagi (AIST), Misaki Ohashi, Yifei Huang, Ryosuke Furuta, Shungo Adachi, Toutai Mitsuyama, Yoichi Sato. AIST + UTokyo + National Cancer Center Research Institute (Japan).
**Source PDF:** `FineBio.pdf` (31 pages, NeurIPS 2024-style preprint, IJCV 2025).
**arXiv:** 2402.00293
**Code/data:** https://github.com/aistairc/FineBio (data hosted on ABCI Cloud Storage, license agreement required)

---

## TL;DR

FineBio is the **most rigorous existing wet-bio video dataset**, but it is *deliberately constrained*:

- **Mock experiments**: real protocol structure, but distilled water instead of reagents and simplified waiting times.
- **Hierarchical annotation only**: every level is a label (`step`, `atomic operation`, `object bounding box`, `manipulation state`), never free-form protocol text.
- **Four perception benchmark tasks** with classical automatic metrics — no LLM-as-judge anywhere.
- **Published participant-level splits**, full reproducibility.

Effectively, FineBio is what LSV would look like if its authors had Computer Vision conference standards rather than Nature Methods–style demo standards. It is the cleanest possible "bottom-up perception" benchmark for lab activity. It does **not** evaluate Protocol Alignment as LSV defines it.

---

## 1. What's in the dataset

### Scale

- **226 trials**, **14.5 hours** of recording, **32 participants** (16M / 16F, each does 5–10 trials)
- **3,541** annotated steps
- **50,659** atomic operations
- **71,548** object bounding boxes across **1,935** sampled frames
- **35 object categories** (left/right hand + 33 lab tools)

### Protocols (7 total, drawn from 4 experiment families)

| #   | Protocol                                          | Steps | Trials |
| --- | ------------------------------------------------- | ----- | ------ |
| 1   | Lysis and recovery of cultured cells (variant 1)  | 9     | 46     |
| 2   | Lysis and recovery of cultured cells (variant 2)  | 12    | 45     |
| 3   | DNA extraction w/ magnetic beads (variant 1)      | 25    | 25     |
| 4   | DNA extraction w/ magnetic beads (variant 2)      | 31    | 26     |
| 5   | Polymerase chain reaction (PCR)                   | 10    | 46     |
| 6   | DNA extraction w/ spin columns (variant 1)        | 18    | 19     |
| 7   | DNA extraction w/ spin columns (variant 2)        | 21    | 20     |

The variant pairs differ by step-count only (different number of wash repetitions, ethanol washes, or centrifuge cycles) — by design, this prevents trivial protocol classification from raw step counts.

### Mock-experiment caveats

- Use distilled water instead of real reagents.
- Simplify or skip waiting times (centrifuge, vortex, magnetic adsorption).
- Replace the PCR machine with a "silver tube rack with distinctive appearance."
- Re-take any trial where a participant made an unrecoverable mistake or got step order wrong.

So this is a *clean* dataset — performance numbers here are an upper bound on what a model could do in a *real* wet-lab setting.

### Camera setup

- **5 fixed third-person cameras** (4000×3000, 30 fps): left-back, left-front, right-back, right-front, top
- **1 head-mounted first-person camera** (3920×2160, 30 fps, wide FOV)
- All synchronized via on-screen QR timecode (<30 ms error)
- Geometrically calibrated (chessboard for third-person, AR markers for first-person)

The benchmark tasks in the paper use **first-person video only**. Third-person streams + multi-view calibration are released for future work.

---

## 2. Annotation hierarchy

The core organizing structure — same shape that LSV inherits:

```
Protocol (7 types, 14.5h)
└── Step (32 categories, average 14.3 sec each)
    └── Atomic Operation (verb × manipulated object × affected object, average 0.91 sec, often instant)
        └── Object bounding boxes + manipulation state (contact / manipulated / affected, per hand)
```

**Steps** are protocol-mandated, ordered, strict. 32 categories total; some shared across protocols.

**Atomic operations** are per-hand and may overlap in time. An operation is `(start, end, verb, manipulated_object, affected_object)`. **10 verbs:** put, take, press, insert, release, open, detach, close, eject, shake. Operations are typically instant (<1 sec), making detection harder than step segmentation.

**Object boxes** are sampled at 1,935 frames (not exhaustive across video) but annotated *fully* in those frames — every object visible, not just active ones. This avoids the bias that "active-only" datasets introduce into background detectors.

**Manipulation state** per object: `contact` (per hand), `manipulated`, `affected`. An object can be manipulated *without* being in contact (e.g. tip inside pipette while pipetting into a tube — the tube is affected, not in direct hand contact).

---

## 3. Splits — published, by participant

| Split   | Participants | Trials | Steps | Atomic ops | Frames | Bboxes |
| ------- | ------------ | ------ | ----- | ---------- | ------ | ------ |
| Train   | 22           | 161    | 2,509 | 36,941     | 1,394  | 50,886 |
| Valid   | 5            | 30     | 482   | 6,458      | 238    | 8,867  |
| Test    | 5            | 35     | 550   | 7,260      | 303    | 11,795 |
| **Total** | **32**     | **226** | **3,541** | **50,659** | **1,935** | **71,548** |

Explicit participant lists in §C.1:

- **Train:** P01, P02, P04, P06, P07, P10–P12, P14, P16–P19, P21–P23, P25–P27, P29–P31
- **Valid:** P05, P09, P15, P24, P32
- **Test:** P03, P08, P13, P20, P28

This matters for our project: **the LabOS-VLM training set used FineBio's training split (22 participants)**. If we evaluate any model on FineBio test (5 participants), it should be unseen by them too — provided LabOS respected the released split, which is plausible but not stated.

A small caveat (§C.3): 4 trials with major mistakes (multi-step missing/redundant) are *removed* from evaluation but kept in the released dataset. 8 trials with minor mistakes are *kept* in evaluation. So the splits are not perfectly clean.

---

## 4. The four benchmark tasks

### Task 1 — Step Segmentation (§4.1)

**Definition.** Per frame, predict step label from 32 categories + background.

**Metrics.** Frame-wise accuracy, segmental edit score, segmental F1@k for k ∈ {10, 25, 50, 75} (k = required temporal IoU between predicted and gold segments to count as a hit).

**Baselines (paper):**

| Model       | Acc  | Edit | F1@10 | F1@25 | F1@50 | F1@75 |
| ----------- | ---- | ---- | ----- | ----- | ----- | ----- |
| MS-TCN++    | 90.2 | 96.7 | 97.4  | 96.7  | 93.5  | **73.4** |
| ASFormer    | 87.2 | 94.8 | 94.2  | 92.7  | 86.5  | **67.0** |

**Read this carefully.** F1@10 is >94% for both models — they get the right step *category* almost always. But F1@75 drops to 67–73% — they get the *exact temporal boundaries* wrong. The dominant failure mode is *boundary precision*, not category mistakes. The paper specifically highlights "add wash buffer" vs "add 70% ethanol" confusion — same atomic actions, different reagent.

### Task 2 — Atomic Operation Detection (§4.2)

**Definition.** Predict a set of `(start, end, verb, manipulated_obj, affected_obj)` tuples for the full video.

**Metric.** mAP at tIoU thresholds {0.3, 0.4, 0.5, 0.6, 0.7}, plus average. An atomic operation is a hit only if all three classes (verb, manip, affected) are correct *and* the temporal IoU exceeds the threshold.

**Baseline (ActionFormer):**

| Target         | mAP@0.3 | 0.4  | 0.5  | 0.6  | 0.7  | Avg  |
| -------------- | ------- | ---- | ---- | ---- | ---- | ---- |
| Atomic op      | 45.2    | 41.7 | 36.5 | 28.4 | 18.7 | **34.1** |
| Verb           | 78.9    | 72.9 | 64.5 | 50.8 | 32.9 | 60.0 |
| Manipulated    | 55.4    | 52.3 | 44.9 | 33.9 | 21.7 | 41.6 |
| Affected       | 65.0    | 60.9 | 54.7 | 43.3 | 28.7 | 50.5 |

So *much* harder than step segmentation. The combinatorial verb × manipulated × affected joint accuracy is what tanks the headline number.

### Task 3 — Object Detection (§4.3)

**Definition.** Detect all 35 object categories in single frames.

**Metric.** Standard COCO AP, AR. Plus AR_manip and AR_affect to measure recall on objects under interaction.

**Baselines:**

| Method            | AP   | AP50 | APS  | APM  | APL  | AR   | AR_manip | AR_affect |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | --------- |
| Deformable DETR   | 53.3 | 77.4 | 11.1 | 30.8 | 59.8 | 62.1 | 55.9     | 51.6      |
| DINO              | **56.1** | 78.5 | 12.6 | 34.2 | 62.5 | 66.6 | 64.0     | 58.8      |

Small objects (APS) are the hard regime. AR on manipulated/affected objects is nearly identical to overall AR — interaction context doesn't damage detection per se.

### Task 4 — Manipulated / Affected Object Detection (§4.4)

**Definition.** Per single frame, jointly detect (a) hand bounding boxes, (b) which object each hand is *manipulating*, (c) which object is *affected* through that manipulation.

**Metric.** Box AP@50, broken into Hand-only / Hand+Manipulated / Hand+Manipulated+Affected.

**Baseline (modified Hand Object Detector, DINO ResNet-50 backbone):**

| Hand       | Hand only | H + Manipulated | H + M + Affected |
| ---------- | --------- | --------------- | ---------------- |
| Left hand  | 96.8      | **6.5**         | **5.9**          |
| Right hand | 94.5      | 22.2            | 10.7             |

**This is the brutal table.** Hand detection is ~95% AP — basically solved. But correctly identifying *what* the hand is manipulating, plus *what* is being affected through it, lands at single digits to low double digits. The paper attributes this to dense object arrangement, occlusion, and the offset-prediction scheme not handling cluttered scenes. Left-hand performance is much worse because right-handers handle small-object work (micro tubes) with their dominant hand.

**This is the hardest perception bottleneck for any lab-VLM.** Whatever Protocol Alignment system we build is downstream of solving (or routing around) this.

---

## 5. What FineBio does *not* do

Worth being explicit, because this is the comparison point with LSV:

- **No free-form protocol-text generation task.** No "watch video → write paragraph" benchmark.
- **No LLM-as-judge.** All metrics are programmatic.
- **No issue-detection task.** Mistake-annotated trials are *excluded* from evaluation, not used as a benchmark.
- **No long-video evaluation.** Average video is 3.9 min; max is single-digit minutes. LSV ranges to 45 min.
- **No XR / glasses / cobot integration.**
- **Mock-not-real.** No actual reagents or chemistry, no wet-lab outcome to validate against.
- **No language interface at all.** This is a pure CV dataset, not a video-language one.

---

## 6. Compute footprint (§A)

The paper trains all baselines on **a single A100 or V100**:

- ActionFormer: ~1 hour
- MS-TCN++: ~2.5 hours
- ASFormer: ~2 days
- Deformable DETR: ~5.5 hours
- DINO: ~1 hour
- Hand-Object Detector: ~15 min

Useful gut-check: training a competitive perception model on FineBio is a few-GPU-hours task, not a multi-day cluster job. If we pursue any FineBio-based baseline, this is well within reach.

---

## 7. Implications for our project

1. **FineBio gives us the cleanest possible "perception sanity check" for any lab-video model.** If a frontier VLM scores poorly on Step Segmentation F1@10 here, it has no business attempting LSV Protocol Alignment.
2. **The published splits are the right ones to use.** We don't have to invent splits the way we did for LSV.
3. **The hardest perception task (manipulated/affected object detection at <25% AP) is the bottleneck under any compound-agent design.** The "compound agent calls a perception sub-model" architecture is only as good as the perception sub-model's ability to answer "what is the hand doing right now?" — and that's at single digits in the published baselines.
4. **The LabOS team almost certainly templated FineBio's annotations into protocol-style prose for SFT** (see our earlier discussion). That conversion is undisclosed in the LabOS paper. We could reconstruct one for ablations.
5. **Comparing FineBio numbers fairly is much easier than LSV** — clean splits, clean metrics, public code on GitHub.
6. **FineBio is not Protocol Alignment.** If the project's North Star metric is Protocol Alignment, FineBio is a *complement*, not a substitute. It probes the perception layer; LSV probes the end-to-end system. Treat them as orthogonal axes if we use both.
