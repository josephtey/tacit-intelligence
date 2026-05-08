# Phase 1 Results — Zero-shot baselines on LSV bench-50

**Eval:** 50 LSV bench videos with gold protocols, sampled with `seed=42`, stratified across XMglass / DJI / Multiview to maximize procedural diversity (18 distinct Operations represented).
**Generators:** GPT-5.5, Gemini 2.5 Pro. Identical prompt, 32 frames uniformly sampled at 1024px, audio stripped, temperature=0, single-shot.
**Judge:** Claude Opus 4.7, text-only (no video), 5-dimension 0–5 rubric + composite mean.
**Run date:** 2026-05-08
**Cost:** ~$5 across all three providers.
**Artifacts:** `runs/predictions/`, `runs/scores/`, `runs/report.md`.

---

## 1. Headline numbers


| Model          | step_coverage | step_hallucination | ordering    | parameter_accuracy | granularity_match | **composite**   |
| -------------- | ------------- | ------------------ | ----------- | ------------------ | ----------------- | --------------- |
| gemini-2.5-pro | 2.50 ± 1.16   | **2.90 ± 1.20**    | 3.84 ± 1.22 | 1.06 ± 0.89        | **2.92 ± 0.88**   | **2.64 ± 0.84** |
| gpt-5.5        | 2.64 ± 1.05   | 1.90 ± 0.89        | 3.70 ± 1.15 | 0.92 ± 0.90        | 2.14 ± 0.53       | 2.26 ± 0.72     |


**Gemini 2.5 Pro wins by 0.38 composite (2.64 vs 2.26).**

For comparison, the LabOS paper reports Gemini 2.5 Pro at **2.86** on their methodology (single 0–5 scalar, undisclosed prompt and split). Our number is 0.22 below — directionally consistent, with our 5-dimension rubric being stricter on hallucination and granularity than a single holistic score would be.

---

## 2. Where the gap actually is

The 0.38 composite delta is driven almost entirely by **two dimensions**:


| Dimension          | Gemini 2.5 Pro | GPT-5.5 | Δ         | What it means                                                                                                  |
| ------------------ | -------------- | ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| step_hallucination | 2.90           | 1.90    | **+1.00** | GPT-5.5 invents many spurious "micro-actions" (tip changes, dial adjustments, walking) that aren't in the gold |
| granularity_match  | 2.92           | 2.14    | **+0.78** | Gemini writes at *protocol level*; GPT over-decomposes a single gold step into 4–6 sub-steps                   |


The other three dimensions are within stdev:


| Dimension          | Gemini 2.5 Pro | GPT-5.5 | Δ     | Read                                           |
| ------------------ | -------------- | ------- | ----- | ---------------------------------------------- |
| step_coverage      | 2.50           | 2.64    | -0.14 | GPT slightly better at catching all gold steps |
| ordering           | 3.84           | 3.70    | +0.14 | Both strong; procedural arc is largely correct |
| parameter_accuracy | 1.06           | 0.92    | +0.14 | **Both fail catastrophically here** — see §3   |


---

## 3. The universal failure: parameter_accuracy ≈ 1.0 / 5

This is the most important finding of Phase 1.

Both frontier models score ~1/5 on `parameter_accuracy`, the dimension that asks: *did you correctly identify reagents, volumes, durations, temperatures, equipment names?* The judge's per-record reasoning shows the same pattern across every video, every protocol type, both models:

- *"describes generic pipetting actions without identifying reagents/volumes"*
- *"misses key reagent steps (loading buffer mix, ladder)"*
- *"omits all numerical parameters"*
- *"reads as a visual description of liquid transfers between colored containers rather than a real protocol"*

The models can see *that* a hand pipettes into a tube. They cannot read *what's on the bottle*, identify the color/state of the liquid, or infer "this is Cas9 plasmid" vs "this is buffer." This is a perception failure, not a reasoning failure — ordering is 3.7+/5 for both, meaning when the model *can* identify what's happening it places it correctly in the procedural arc.

**Implication:** any Phase 2 system that doesn't address parameter perception is leaving ~1 full composite point on the table. This is the highest-ROI lever in the whole project.

---

## 4. Per-model failure signatures

### GPT-5.5 — over-decomposition + hallucination

GPT-5.5's outputs read like dense visual narration. A gold step like *"Add 1 mL Opti-MEM to a sterile 1.5 mL EP tube"* becomes 4–6 predicted steps:

> 1. Arrange labeled microcentrifuge tubes in a green tube rack on the lab bench and open the caps of the tubes to be used.
> 2. Attach a disposable pipette tip to a Transferpette S adjustable micropipette from the tip box.
> 3. Pipette liquid reagent into an open labeled microcentrifuge tube held in the gloved hand.
> 4. Place the filled microcentrifuge tube back into the green rack with the cap open.

Each individual sentence is *correct* visually but *wrong as protocol prose*. The rubric penalizes this twice — once on `granularity_match` (decomposition mismatch) and once on `step_hallucination` (the trivial sub-actions count as spurious steps relative to gold).

### Gemini 2.5 Pro — protocol-level prose, but still parameter-blind

Gemini writes at the right level of abstraction:

> 1. Place several microcentrifuge tubes into a green tube rack and open their caps.
> 2. Using a Transferpette S micropipette, transfer a liquid into the open microcentrifuge tubes.
> 3. Using a second, smaller-volume micropipette, transfer another liquid into the microcentrifuge tubes.

Same procedure, far fewer steps, no fabricated handling. But notice "a liquid", "another liquid" — Gemini doesn't know they're Opti-MEM, Cas9 plasmid, gRNA either. Same parameter blindness, just expressed more concisely.

---

## 5. Distribution and outliers

- **Best record (Gemini):** XM_063 (PCR setup, composite 4.0+ range)
- **Best record (GPT-5.5):** MV_058 (composite 3.8 — full coverage with parameters, just over-decomposed)
- **Worst records:** Empty outputs from both models (DJI-037 from Gemini at 0.0, DJI-014 from GPT-5.5 at 0.0). Worth investigating — likely a frame-extraction edge case for very short clips.
- Most predictions cluster in 2.0–3.0 composite, consistent with LabOS's reported regime.

---

## 6. Phase 2 implications

In priority order:

1. **Parameter perception is the load-bearing lever.** Both models are stuck at ~1/5 on this dimension. Any of the following could realistically move it to 2–3/5:
  - OCR overlay on bottle labels and pipette readouts (cheap, easy, immediate)
  - Object-detection auxiliary head trained on FineBio's hand-object annotations (~14.5 h of supervision available)
  - Equipment + reagent vocabulary injected at prompt time (lab-specific named entity priors)
  - Fine-tuned small VLM for the perception layer feeding a frontier reasoner
2. **GPT-5.5 specifically benefits from prompt-level discipline.** Telling it "write at protocol level, not visual narration level" may close the granularity + hallucination gap without any architectural change. Worth ablating before assuming Gemini is a better base.
3. **Ordering is solved.** ~3.7+/5 for both models. Don't spend Phase 2 effort here.
4. **Compound-agent vs fine-tune is a real choice now.** With a clear bottleneck (perception, specifically parameters), the architecture question reduces to: do we close the gap with (a) a small dedicated perception model + frontier reasoner, or (b) a single fine-tuned multimodal model? Each has distinct failure modes worth comparing head-to-head.

---

## 7. Methodological footnotes

- **Judge:** Claude Opus 4.7, text-only, sees predicted + gold protocols only (no video). Avoids the multimodal-judge confound where a vision-capable judge might agree with same-family hallucinations.
- **Prompt:** Single shared template (`eval/prompts/generation.txt`), byte-identical for both providers.
- **Frame budget:** 32 uniform frames at 1024px JPEG, audio stripped, audio narration not transcribed.
- **Reproducibility:** Manifest is deterministic (`seed=42`); predictions and scores cache per-`(model, slice_id, prompt_hash, frame_count)` so reruns of unchanged inputs are free.
- **Comparison to LabOS:** Their Gemini 2.5 Pro = 2.86 vs ours = 2.64. We did not reproduce their exact methodology (judge prompt unavailable, split unavailable, frame budget unspecified). Numbers are directionally aligned but not strictly comparable.

