# Closing the frontier-VLM gap on wet-lab Protocol Alignment

Frontier VLMs cannot yet watch a lab video and faithfully reconstruct what happened. On **Protocol Alignment** in LabSuperVision (Cong et al. 2025, [arXiv:2510.14861](https://arxiv.org/abs/2510.14861)) — a benchmark of ~200 egocentric lab videos — GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-7B all score 2.0–2.86 / 5; only LabOS's in-house 235B VLM closes the gap. This project aims to build a system that meaningfully beats the strongest frontier baseline on that metric.

## Goals

**North Star:** beat the strongest zero-shot frontier baseline on Protocol Alignment (LabSuperVision), measured by the published 0–5 rubric. The form of the system is deliberately left open — small fine-tuned VLM, compound agent, or hybrid.

Four sub-questions structure the work:

1. **Where does the frontier actually break?** Build a structured failure taxonomy across protocol types and error classes. The diagnosis drives every later design decision.
2. **Train, compose, or hybrid?** Measure empirically which shape of system best moves the rubric — the central question for any AI-for-science company deciding between training its own models and building on frontier ones.
3. **What signal actually matters?** Ablate across modalities (video, audio narration, object detections, equipment state) and data regimes (training scale, context length, tool calls, reasoning depth) to find which signals are worth the cost of capturing.
4. **What does utility look like beyond the rubric?** Show outputs to 2–3 wet-lab scientists on real procedures and collect qualitative signal where the benchmark and human judgment diverge.

## Experiments and success criteria

| # | Experiment | Success criterion |
|---|---|---|
| North Star | Best system vs. strongest zero-shot frontier baseline on a held-out LSV bench slice. | Composite score ≥ +0.3 over the best frontier baseline on the held-out slice. |
| Q1 — Frontier failure modes | Reproducible zero-shot harness over GPT-5.5 and Gemini 2.5 Pro on a 50-video bench slice, scored by a multi-dimensional rubric (`step_coverage`, `step_hallucination`, `ordering`, `parameter_accuracy`, `granularity_match`). | ≥ 3 named failure modes, each tied to a sub-score and supported by ≥ 5 example slices. |
| Q2 — Train / compose / hybrid | Three head-to-head systems on the same slice: (a) LoRA-tuned small open-weight VLM, (b) compound frontier agent with visual tool use + decomposition + memory, (c) hybrid (small perception model → larger reasoner). | One system wins by ≥ 0.2 composite, *and* the win is explained by the Q1 taxonomy (i.e. it closes the specific sub-scores Q1 identified as the gap). |
| Q3 — What signal matters | Ablations on the winning system: video-only vs. audio-only, ± object detections, frame budget (8/16/32/64), reasoning depth, training-data size. | Quantified Δ-composite per axis, with a clear worth-the-cost / not-worth-the-cost verdict for each signal. |
| Q4 — Utility beyond the rubric | Show outputs from the best system to 2–3 wet-lab scientists on procedures from their own work; structured qualitative interview. | Documented list of failure/utility modes the rubric misses, plus ≥ 1 concrete benchmark-improvement proposal grounded in the interviews. |

## Phase 1 status: zero-shot eval harness

Phase 1 addresses Q1 — a clean, reproducible Protocol Alignment harness on a 50-video subset of LSV (`Scene == "bench"`, with associated gold protocols).

- **Generators:** `gpt-5.5`, `gemini-2.5-pro` (modular — adding a third is one file)
- **Judge:** `claude-opus-4-7` with a published, multi-dimensional rubric
- **Inference recipe:** 32 frames uniformly sampled at 1024px, audio stripped, identical prompt across providers
- **Predicted total cost:** ~$17 for the full 50-video × 2-generator × 1-judge run

## Repo layout

```
data/lsv/                       # symlink to LSV dataset (gitignored)
readings/                       # paper distillations
  LabOS.md                      # source paper + LSV benchmark
  FineBio.md                    # closest sibling dataset (perception-only)
eval/
  manifest.py                   # build the 50-video manifest
  frames.py                     # uniform frame extraction (ffmpeg)
  generators/                   # ProtocolGenerator ABC + OpenAI + Gemini
  judge/                        # ProtocolJudge ABC + Claude
  prompts/
    generation.txt              # shared prompt for all generators
    rubric.txt                  # judge rubric with 0–5 anchors
  run.py                        # generation entry point (--shard i/n)
  score.py                      # scoring entry point
  report.py                     # aggregate report
scripts/
  eval_lsv.sbatch               # 4-task SLURM array (cpu partition)
  eval_lsv_report.sbatch        # dependent aggregate-report job
  submit_eval.sh                # convenience wrapper
  progress.sh                   # one-shot status snapshot
  download_lsv.py               # resilient HF dataset downloader
tests/                          # pytest harness for non-API components
runs/                           # all eval outputs, gitignored
  manifests/bench_50_seed42.json
  predictions/{model}/{slice_id}__{hash}.json
  scores/{judge}/{model}/{slice_id}__{hash}.json
  frames/{slice_id}/*.jpg
  slurm_logs/
  report.md                     # final aggregate
```

## Running

One-time setup:

```bash
cp .env.example .env             # then fill in the three API keys
pip install -r requirements.txt
python -m eval.manifest          # builds runs/manifests/bench_50_seed42.json
pytest tests/ -v                 # 11 tests, all non-API
```

Smoke test (1 video, ~$0.50, ~30 s):

```bash
python -m eval.run --models gpt-5.5 gemini-2.5-pro --limit 1
python -m eval.score --judge claude-opus-4-7 --models gpt-5.5 gemini-2.5-pro --limit 1
```

Full eval on the cluster:

```bash
bash scripts/submit_eval.sh      # array job + dependent report
bash scripts/progress.sh         # status snapshot anytime
```

The pipeline is fully cached (per-`(model, slice_id, prompt_hash, frame_count)`), so reruns with unchanged inputs are free.

## Reproducibility commitments

The LabOS paper has notable methodological gaps — undisclosed inference recipe, unpublished judge prompt, undisclosed split, metric switching between baseline and headline tables. This harness commits to choices that make our numbers reproducible even where LabOS's are not:

- **Defined split.** 50 videos sampled deterministically (`seed=42`) from the bench-with-protocol pool of 142, stratified across XMglass / DJI / Multiview to maximize procedural diversity.
- **Frozen frame budget.** 32 uniform frames at 1024px, audio stripped, applied identically to every generator. Frame budget is itself an axis we'll ablate later (Q3).
- **Single shared prompt.** Byte-identical for OpenAI and Gemini. Prompt hash is part of the cache key so any change auto-invalidates old predictions.
- **Multi-dimensional rubric.** Judge produces five sub-scores plus a composite — the LabOS paper reports a single 0–5 scalar with no decomposition.
- **Text-only judge.** Claude sees the predicted and gold protocols, not the video. Avoids the multimodal-judge confound where a vision-capable judge could "agree" with a same-family generator's hallucinations.

## References

- LabOS paper (LSV benchmark): [arXiv:2510.14861](https://arxiv.org/abs/2510.14861)
- LSV dataset: [labos1/LSV on Hugging Face](https://huggingface.co/datasets/labos1/LSV) (CC-BY-NC-4.0)
- FineBio paper (sibling perception dataset): [arXiv:2402.00293](https://arxiv.org/abs/2402.00293)
