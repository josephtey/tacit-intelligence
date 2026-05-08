# Closing the frontier-VLM gap on wet-lab Protocol Alignment

Frontier VLMs cannot yet watch a lab video and faithfully reconstruct what happened. On **Protocol Alignment** in LabSuperVision (Cong et al. 2025, [arXiv:2510.14861](https://arxiv.org/abs/2510.14861)) — a benchmark of ~200 egocentric lab videos — GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-7B all score 2.0–2.86 / 5; only LabOS's in-house 235B VLM closes the gap. This project builds a system that meaningfully beats the strongest frontier baseline on that metric.

---

## The task: Protocol Alignment on LabSuperVision

**LabSuperVision (LSV)** is a corpus of ~200 egocentric lab videos collected by 7 researchers via head-mounted cameras and smart glasses while running real wet-bio experiments — PCR setup, transformations, CRISPR/Cas9 delivery, gel electrophoresis, etc. Each video is paired with an expert-authored **gold-standard protocol** (the textual SOP that should describe what the researcher did).

**Protocol Alignment** asks: *given the video, can a model produce a stepwise protocol that matches the gold?* Concretely:

> **Input:** a sequence of frames sampled from one experimental session.
> **Output:** numbered, free-form text describing what was performed, with reagents, volumes, durations, equipment.
> **Score:** 0–5, judged against the gold protocol.

Every base model (GPT-5.5, Gemini 2.5 Pro, etc.) in this project receives the **identical prompt** and **identical 32-frame budget**, then writes a stepwise protocol in their own voice. The judge compares predicted vs. gold text and produces structured scores.

---

## How we measure goodness: the rubric

We score every prediction on a **5-dimension rubric**, each dimension 0–5, with a composite mean. The full anchored rubric is at `eval/prompts/rubric.txt`; the dimensions are:

| Dimension              | What it asks                                                                                  | High score means                                          |
| ---------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `step_coverage`        | Of the gold steps, how many appear in the prediction? *(recall)*                              | All gold steps captured                                   |
| `step_hallucination`   | Of the predicted steps, how many are spurious vs. mapping to a gold step? *(precision)*       | Few hallucinated/fabricated steps                         |
| `ordering`             | Are captured steps in the gold's sequence?                                                    | Correct procedural arc                                    |
| `parameter_accuracy`   | Volumes, durations, temperatures, equipment, reagent names — match the gold?                  | Specific values right                                     |
| `granularity_match`    | Is the level of decomposition similar to gold? Not too coarse, not too fine.                  | Same step density as gold                                 |

Each dimension has explicit 0/1/2/3/4/5 anchors (e.g. *step_coverage = 4 means "one gold step missing or merged"*). The judge (Claude Opus 4.7, text-only — does not see the video) returns structured JSON; reasoning per-dimension is logged for auditing. The rubric is a single defensible operationalization of "is the predicted protocol a faithful reconstruction of what happened" — not the only one possible, but ours is published and the prompt template is part of the cache key, so any change auto-invalidates old scores.

---

## Goals (Q1)

**North Star:** beat the strongest zero-shot frontier baseline on Protocol Alignment (LSV bench slice), measured by the 5-dimension rubric above.

**The bet:** a compound system, not a fine-tuned monolith. Specifically:

- **A small perception model trained on FineBio** ([arXiv:2402.00293](https://arxiv.org/abs/2402.00293)) — 14.5 h of hierarchically-annotated wet-bio video with bounding boxes, hand-object states, atomic operations, and step labels. FineBio's published baselines for object detection (~56 AP) and step segmentation (~97% F1@10) are far stronger than what frontier VLMs can do natively.
- **An LLM reasoner** (frontier — GPT-5.5, Gemini, etc.) that uses the small perception model's outputs as auxiliary signal to construct the protocol. The reasoner doesn't have to *see* labels on bottles; the perception model does, and feeds the identification as text.
- **Post-training the reasoner to use the perception tools well.** A frontier LLM given a tool-use harness around object detections + step segments isn't guaranteed to use them effectively zero-shot. SFT or RL on synthetic (perception trace → correct protocol) pairs may be needed to teach the reasoner *when* and *how* to invoke perception, weight conflicting signals, and avoid over-reliance on tool calls that come back uncertain.

The thesis: **the bottleneck is perception, not reasoning** (Phase 1 results below support this). Frontier reasoners are already strong at constructing protocol prose given the right facts; what they cannot do is *see* "this bottle is labeled Opti-MEM." Outsourcing perception to a small specialized model and post-training the reasoner to consume it should close the gap more cheaply than a 235B-parameter monolithic VLM.

Three sub-questions structure the work along the way:

1. **Where does the frontier actually break?** Build a structured failure taxonomy across protocol types and rubric dimensions. *(Answered by Phase 1, see results.)*
2. **What signal actually matters?** Ablate across modalities (object detection, step segmentation, audio, frame budget) and post-training regimes (zero-shot tool use, SFT on perception traces, RL).
3. **What does utility look like beyond the rubric?** Show outputs to 2–3 wet-lab scientists on procedures from their own work; collect qualitative signal where the benchmark and human judgment diverge.

---

## Experiments and success criteria (Q2)

### Done — Phase 1: zero-shot baselines on LSV bench-50

A reproducible zero-shot harness over GPT-5.5 and Gemini 2.5 Pro on a curated 50-video slice of LSV.

**How the eval was constructed:**

- Filter LSV to `Scene == "bench"` clips with associated protocol text (142 candidates across XMglass / DJI / Multiview).
- Sample 50 deterministically (`seed=42`), stratified 20 / 20 / 10 across the three camera setups, deduplicating Multiview's time-aligned multi-phone clips so different *experimental moments* — not different camera angles of the same moment — get sampled.
- 18 distinct Operations represented (PCR, transformation, Cas9 delivery, E-gel loading, serial dilution, etc.).
- Manifest is frozen at `runs/manifests/bench_50_seed42.json`.

**Results** (judge: `claude-opus-4-7`, n = 50 per model):

| Model            | step_coverage | step_hallucination | ordering    | parameter_accuracy | granularity_match | **composite** |
| ---------------- | ------------- | ------------------ | ----------- | ------------------ | ----------------- | ------------- |
| gemini-2.5-pro   | 2.50 ± 1.16   | **2.90 ± 1.20**    | 3.84 ± 1.22 | 1.06 ± 0.89        | **2.92 ± 0.88**   | **2.64 ± 0.84** |
| gpt-5.5          | 2.64 ± 1.05   | 1.90 ± 0.89        | 3.70 ± 1.15 | 0.92 ± 0.90        | 2.14 ± 0.53       | 2.26 ± 0.72   |

**Key finding: parameter_accuracy ≈ 1 / 5 across both models.** Both frontier VLMs cannot read bottle labels, identify reagents from frames, or recover specific volumes/durations — but they get ordering ~3.7+/5, meaning the procedural reasoning is intact when they *can* identify what is happening. The bottleneck is perception, specifically of named entities and numerical parameters. This directly motivates the FineBio-perception bet. Full analysis at `readings/results.md`.

**Success criterion (Phase 1):** ≥ 3 named failure modes tied to a rubric dimension and supported by ≥ 5 example slices. ✅ Met — see `readings/results.md` §3–§4 (parameter blindness, GPT over-decomposition, mutual hallucination).

### Planned — Phase 2: compound system vs. baseline

Build the FineBio-perception + LLM-reasoner system in two phases:

**2a — Tool-use harness (no training).** Wrap FineBio-trained object detection (DINO ~78 AP50) + step segmentation (MS-TCN++ ~97 F1@10) as tool calls. Frontier LLM zero-shot consumes detections and writes the protocol. Compare to Phase 1 baseline.

- *Success:* composite ≥ +0.3 over the strongest Phase 1 baseline, *with the gain explained by parameter_accuracy and step_coverage* (i.e. closing the specific dimensions Phase 1 identified).

**2b — Post-trained reasoner.** If 2a underperforms because the LLM doesn't use the tools well, SFT or RL the reasoner on (perception trace, gold protocol) pairs derived from FineBio + LSV training splits. Possibly a small reasoning model rather than frontier API.

- *Success:* ≥ +0.2 composite over 2a, *and* ablations show the gain is from learned tool-use behavior rather than memorization of LSV protocols.

**2c — Ablations.** Frame budget (8 / 16 / 32 / 64), object detection on/off, step segmentation on/off, audio narration via Whisper on/off. Quantified Δ-composite per axis.

- *Success:* clear worth-the-cost / not-worth-the-cost verdict for each signal.

### Planned — Phase 3: utility validation

Show the best system's outputs to 2–3 wet-lab scientists on procedures from their own work; structured qualitative interview.

- *Success:* documented list of failure/utility modes the rubric misses, plus ≥ 1 concrete benchmark-improvement proposal grounded in the interviews.

---

## Methodology (brief)

- **Generators:** GPT-5.5, Gemini 2.5 Pro. Identical prompt template, 32 uniform frames at 1024px, audio stripped, temperature 0.
- **Judge:** Claude Opus 4.7, text-only — sees predicted + gold protocol text only, *not* the video. Avoids same-family multimodal confound.
- **Rubric:** 5 dimensions × 0–5, anchored, published at `eval/prompts/rubric.txt`. Composite = mean.
- **Caching:** every prediction and score keyed on `(model, slice_id, prompt_hash, frame_count)`. Reruns of unchanged inputs are free.
- **Modular:** adding a new generator is one file under `eval/generators/`; new judge under `eval/judge/`. Intentionally swappable for fine-tuned variants.

The LabOS paper has notable methodological gaps (undisclosed inference recipe, unpublished judge prompt, undisclosed split, metric switching between tables). This harness commits to fully published choices so our numbers are reproducible.

---

## Repo layout

```
data/lsv/                         # symlink to LSV dataset (gitignored)
readings/                         # paper distillations + results writeup
  LabOS.md                        # source paper + LSV benchmark
  FineBio.md                      # closest sibling dataset (perception-only)
  results.md                      # Phase 1 analysis (read this for the findings)
eval/
  manifest.py                     # build the 50-video manifest
  frames.py                       # uniform frame extraction (ffmpeg)
  generators/                     # ProtocolGenerator ABC + OpenAI + Gemini
  judge/                          # ProtocolJudge ABC + Claude
  prompts/
    generation.txt                # shared prompt for all generators
    rubric.txt                    # judge rubric with 0–5 anchors
  run.py                          # generation entry point (--shard i/n)
  score.py                        # scoring entry point
  report.py                       # aggregate report
scripts/
  eval_lsv.sbatch                 # 4-task SLURM array (cpu partition)
  eval_lsv_score.sbatch           # scoring-only sbatch
  eval_lsv_gemini.sbatch          # gemini gen+score+report
  submit_eval.sh                  # convenience wrapper
  progress.sh                     # one-shot status snapshot
  download_lsv.py                 # resilient HF dataset downloader
tests/                            # pytest harness for non-API components
runs/                             # all eval outputs, gitignored
  manifests/bench_50_seed42.json
  predictions/{model}/{slice_id}__{hash}.json
  scores/{judge}/{model}/{slice_id}__{hash}.json
  frames/{slice_id}/*.jpg
  slurm_logs/
  report.md                       # final aggregate
```

## Running

```bash
cp .env.example .env             # fill in 3 API keys (OpenAI, Google, Anthropic)
pip install -r requirements.txt
python -m eval.manifest          # builds runs/manifests/bench_50_seed42.json
pytest tests/ -v                 # 11 tests, all non-API
bash scripts/submit_eval.sh      # full eval on cluster
bash scripts/progress.sh         # status snapshot anytime
```

Smoke test (1 video, ~$0.50):

```bash
python -m eval.run --models gpt-5.5 gemini-2.5-pro --limit 1
python -m eval.score --judge claude-opus-4-7 --models gpt-5.5 gemini-2.5-pro --limit 1
```

## References

- LabOS paper (LSV benchmark): [arXiv:2510.14861](https://arxiv.org/abs/2510.14861)
- LSV dataset: [labos1/LSV on Hugging Face](https://huggingface.co/datasets/labos1/LSV) (CC-BY-NC-4.0)
- FineBio paper (sibling perception dataset): [arXiv:2402.00293](https://arxiv.org/abs/2402.00293)
- Phase 1 results writeup: `readings/results.md`
