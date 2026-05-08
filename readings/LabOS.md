# LabOS: The AI-XR Co-Scientist That Sees and Works With Humans

**Authors (lead):** Le Cong (Stanford), Mengdi Wang (Princeton). Stanford Pathology/Genetics + Princeton AI Lab + NVIDIA + Oregon State + UW.
**Source PDF:** `LabOS.pdf` (19 pages, 2025).
**Companion benchmark paper:** Cong et al. 2025, *LabSuperVision*, arXiv:2510.14861.

---

## TL;DR

LabOS is a "co-scientist" platform that fuses three things:

1. **A self-evolving multi-agent system (STELLA-based)** for the *digital* lab — Manager, Developer, Critic, Tool-Creation agents over a shared "Tool Ocean."
2. **A lab-specialized VLM** that interprets egocentric lab video so AI can "see" experiments.
3. **An AR/XR glasses + cobot stack** for real-time human-in-the-loop guidance and hand-off in the *physical* lab.

The paper's two big technical claims:

- On biomedical reasoning, the digital agent sets new SoTA on HLE-Biomed (~32%), LAB-Bench DBQA (~61%), LAB-Bench LitQA (~65%) — beating the next-best by up to 8%.
- On lab video understanding (their new **LabSuperVision / LSV** benchmark), all frontier VLMs flop (best baseline 2.86/5). Their post-trained LabOS-VLM family (7B → 235B) closes the gap; the 235B model exceeds 90% error-detection accuracy on a held-out subset.

The ambition is bigger than any of the components: turn a real lab into a perceivable, operable, auditable environment so AI can *participate* in experiments, not just propose them.

---

## 1. Problem framing

- AI has eaten the *computational* side of science (AlphaFold, simulation, design). The bottleneck is now *physical* execution — perception, coordination, tacit skill.
- "Agentic AI" today is digital-only. Lab automation today is rule-based and brittle.
- LabOS's bet: a unified human–AI loop that spans hypothesis → design → physical execution → documentation, mediated by XR glasses and a VLM that watches what the scientist does.

The reproducibility crisis is reframed as a *legibility* problem: most of what skilled experimenters do is never written down. Capturing it requires a model that can watch a real bench session and faithfully reconstruct the procedure.

---

## 2. System architecture

### 2.1 Digital (dry) lab — the agent

Built on / extends **STELLA** (Jin et al., arXiv:2507.02004), a multi-agent reasoning loop:

- **Manager / Planner Agent:** decomposes a research goal into modules (candidates, reagents, procedures, materials, instrument settings, QC checkpoints).
- **Developer Agent:** writes and runs Python for bioinformatics analyses.
- **Critic Agent:** evaluates intermediate results, drives the iterative loop.
- **Tool-Creation Agent:** identifies, tests, integrates new analytical tools, APIs, databases. Drops them into a shared **Tool Ocean**.
- **Template Library:** stores successful reasoning workflows so the system generalizes from prior solutions.

Self-improvement comes from two sources: the Tool Ocean (capability accretion) and the Template Library (workflow accretion). The paper presents inference-time scaling curves as evidence the design actually self-evolves.

### 2.2 Physical (wet) lab — XR glasses + VLM

- **Hardware:** AR/XR glasses (Viture). <85g, 2+ hr battery via neckband, 1200+ Nits brightness, 6DoF + hand-gesture tracking. They also tested VR/XR headsets but chose AR for ergonomics.
- **Streaming loop:** glasses stream first-person video (4 fps for VLM input) to a local GPU server (or cloud) in 5–10 s segments. Server runs the LabOS-VLM, returns structured JSON. Unity/Android app on the glasses parses the JSON and renders inline guidance, error prompts, and audio feedback.
- **What it does live:** stepwise protocol overlay, action verification against the gold-standard protocol, error/deviation detection, context-aware hints.
- **Documentation:** every stream is timestamped + tagged with metadata so each run is auto-logged.

### 2.3 3D/4D world model

For spatial grounding they compose off-the-shelf models:

- **MapAnything** (Keetha et al. 2025, arXiv:2509.13414): feed-forward metric 3D reconstruction from multi-view + egocentric.
- **3D Gaussian Splatting** + **4D-LangSplat** (Li et al. 2025): time-aware, semantically indexable scene reconstruction.
- **HAPTIC** for hand tracking; **MegaPose** for 6-DoF object pose from CAD; **HORT** for scale alignment between hand and object trajectories.

Output: a digital twin that supports replay, what-if simulation, and training data generation.

### 2.4 Cobot module

xArm + gripper + Intel RealSense, controlled by an open-vocabulary perception stack. Hand-off between human and arm is mediated by the VLM. Demos: vortexing, 96-well plate ops, tube handling on incubator/shaker. Proof-of-concept only.

---

## 3. LabSuperVision (LSV) — the benchmark

This is the piece most relevant to our project.

### 3.1 Data

- **>200 video sessions**, typically 2–10 min (up to 45 min), from 7 researchers.
- Captured via **XR glasses or action cameras** in real labs — bench, tissue culture, instrument bays — including movement between rooms.
- Settings: biomedical and material-science labs.
- Each session is paired with an **expert-generated gold-standard protocol**.

### 3.2 Annotations

Five expert annotators per video produce:

1. **Step segments** with start/stop timestamps aligned to the gold-standard protocol.
2. **Error / issue events** typed by category — sterile breach, step mismatch, timing deviation, etc.
3. **Critical parameters, materials, reagents** where applicable.

### 3.3 Tasks and rubric

Two evaluation tasks:

- **Protocol Alignment.** Given a video, generate a stepwise protocol describing actions and parameters. Compared against the gold-standard protocol.
- **Issue Identification.** Given a video, flag deviations and handling errors. Compared against the expert error labels.

Scoring is **0–5** (5 = perfect). Two parallel judges:

- **n = 5 human experts** scoring against a published rubric (rubric in the supplement).
- **GPT-5 as auto-judge** using the same rubric.

### 3.4 Baseline numbers (from Fig. 2c)

Models evaluated zero-shot:

| Model              | Protocol Alignment | Issue Identification |
| ------------------ | ------------------ | -------------------- |
| Gemini 2.5 Pro     | **2.86**           | ~2.0                 |
| NVIDIA Cosmos-1    | 2.24               | —                    |
| GPT-4o             | ~2.0–2.2           | ~2.0                 |
| Qwen 2.5-VL-7B     | low (~2.0)         | low                  |

The takeaway the paper hammers: **frontier VLMs partially handle protocol alignment and badly fail at error detection.** This is the gap our project targets.

---

## 4. LabOS-VLM — the trained model

### 4.1 Training data

- **FineBio** (Yagi et al. 2025): expert-annotated physical-lab videos with hierarchical annotations.
- **JoVE**: standardized procedure videos (third-party scientific video corpus).
- **LSV** itself.

Split: **80 / 10 / 10** train / val / held-out test.

### 4.2 Training recipe

- **Base model:** Qwen-VL family.
- **Stage 1 — SFT with LoRA** on paired video–text examples.
- **Stage 2 — RL fine-tune with GRPO + LoRA.** The model rolls out a group of candidate responses per prompt; rewards are *rule-based* and target (a) procedural accuracy, (b) safety compliance, (c) detail level, plus a within-group relative term that favors expert-consistent reasoning.
- Released sizes: **7B, 32B, 72B, 235B**.

### 4.3 Results

- All LabOS-VLM sizes beat the base Qwen-VL on protocol generation quality and error detection on the held-out set.
- **LabOS-VLM-235B** exceeds 90% accuracy on error detection — beating Claude Opus 4.1, GPT-5, and Gemini 2.5 Pro on every reported metric.
- Qualitative: on real CRISPR transfection videos it correctly distinguished correct vs. incorrect operations and pinpointed two distinct procedural errors.

---

## 5. Use cases (validation studies)

1. **NK immunotherapy target discovery (CRISPRa screen, A375 melanoma).** LabOS analyzed the screen, dynamically re-ranked genes via iterative pathway enrichment, nominated **CEACAM6**, did automated TCGA survival analysis, and the wet-lab CRISPRa validation confirmed CEACAM6 increases NK resistance.
2. **Cell-engineering co-pilot for CEACAM6 validation.** Junior scientists wearing glasses hit >80% CEACAM6 over-expression efficiency by following AI guidance recorded from expert sessions — a tacit-knowledge-transfer demo.
3. **Cell-fusion mechanism discovery.** LabOS proposed **ITSN1** as a cell-fusion regulator. Wet-lab CRISPRi + FAST-induced fusion assay in U2OS cells confirmed ITSN1 knockdown reduces fusion.
4. **Stem-cell engineering** (lentiviral transduction in iPSCs): expert workflows auto-digitized, replayed for novice training.

---

## 6. What's *not* in the paper

Worth keeping in mind when planning our work:

- **No public release** of LabOS-VLM weights stated in this paper (LSV release status: implied via the standalone arXiv:2510.14861).
- **No detailed eval protocol** for LSV scoring beyond "five experts + GPT-5 judge with a rubric in the supplement" — the rubric specifics need to be pulled from the LSV paper / supplement.
- **Sample sizes per task** (how many videos per scoring split, how many per protocol type) are not broken out in the main text.
- **Inter-rater agreement** between the five human experts isn't reported.
- **Compute budget** for SFT + GRPO on the 235B model isn't disclosed.
- **Failure modes** of the baselines aren't dissected — only aggregate scores. This is precisely the gap our Phase 1 failure taxonomy is meant to fill.

---

## 7. Key references to chase next

| Citation                                | Why it matters for us                                        |
| --------------------------------------- | ------------------------------------------------------------ |
| LSV paper (arXiv:2510.14861)            | Full benchmark spec, rubric, splits, leaderboard.            |
| STELLA (arXiv:2507.02004, Jin et al.)   | The dry-lab agent we'd reproduce / build on.                 |
| Alita (arXiv:2505.20286, Qiu et al.)    | Generalist self-evolving agent that STELLA extends.          |
| FineBio (Yagi et al. IJCV 2025)         | Largest pretraining-shaped lab video corpus we can pull.     |
| GRPO (DeepSeek, arXiv:2504.09374)       | RL recipe for the SFT→RL stage.                              |
| Qwen-VL                                 | Base model — open weights, what we'd LoRA on.                |
| LAB-Bench (arXiv:2407.10362)            | Companion text-only biomedical benchmark.                    |
| MapAnything / 4D-LangSplat / HAPTIC etc | Only relevant if we touch 3D world-modeling. Likely defer.   |
