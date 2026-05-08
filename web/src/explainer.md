# How frontier VLMs turn a lab video into a protocol

The **Protocol Alignment** task on LSV asks a model to watch a 2–10 minute (sometimes 45-minute) egocentric lab video and write a stepwise, parameterized protocol — actions, reagents, volumes, durations, equipment settings — that an expert grader can compare against a gold protocol on a 0–5 rubric.

There is no single "feed video to model" interface. What "video input" actually means depends on the model.

## Model interfaces differ wildly

| Model                | Native video?       | What the caller actually sends                                         |
| -------------------- | ------------------- | ---------------------------------------------------------------------- |
| **Gemini 2.5 Pro**   | Yes, via Files API. | Upload the raw `.mp4`; the model samples frames (~1 fps default) + reads the audio track. |
| **Qwen 2.5-VL**      | Yes.                | Native video tensor input; internal frame sampler at a configured fps. |
| **GPT-4o / GPT-5**   | No.                 | Caller extracts N frames as images and sends them as an image sequence. Common: 8 / 16 / 32 / 64 uniformly sampled. |
| **Claude Opus 4.6**  | No.                 | Same — image sequence. Audio (if used) goes through a separate transcription step. |

Two consequences worth dwelling on:

1. **A single "GPT-4o vs Gemini" comparison hides a frame-budget choice.** Running GPT-4o at 8 vs 64 frames on the same video can swing scores by ~1 rubric point. Any honest baseline number has to specify the frame-extraction policy.
2. **Audio is a hidden modality.** Many lab videos contain narration ("now I'm aspirating 200 microliters into well B3"). Gemini reads the audio natively; GPT-4o doesn't unless you transcribe it (e.g., Whisper) and feed the text alongside the frames. A "fair" baseline has to decide whether audio is in or out.

## What a typical pipeline looks like

A frontier-model baseline for Protocol Alignment is rarely just "one API call." A realistic pipeline:

1. **Pre-process video.** Decide on `fps`, `n_frames`, sampling strategy (uniform, keyframe, scene-change). For long videos (>10 min), decide on chunking (sliding window, chapter splits).
2. **(Optional) Transcribe audio** — Whisper or a similar model — if the target VLM doesn't ingest audio natively.
3. **Prompt the VLM** with: the frames (or video file), optionally the transcript, and a system prompt asking for a structured protocol (numbered steps, parameters in brackets, etc.).
4. **(Optional) Refine** — a second pass that asks the model to fix obvious gaps, normalize units, or merge overlapping steps.

Each step is a knob. The "what signal actually matters" question that the LSV paper deliberately leaves open is exactly the question of which knobs move the score.

## How the rubric judges the output

LSV uses two parallel judges: **5 human experts** and **GPT-5 as auto-judge**, both scoring 0–5 against a published rubric (the rubric lives in the paper's supplement and is not open-sourced as code). Roughly, a "good" protocol output has to capture:

- **Step granularity.** Right level of decomposition — not too coarse ("did the experiment"), not too fine ("moved finger 1 cm to the right").
- **Parameters.** Volumes, concentrations, durations, RPMs, temperatures, reagent identities. Many of these are partly visible (label on a tube, digital readout on a vortexer) and partly inferred from convention.
- **Action verbs.** "Vortex" vs "mix" vs "swirl" vs "flick" carry different semantics in a chemistry protocol.
- **Order and dependencies.** Reagents prepared before they're used; centrifuge spins before tubes are removed.

The judge sees the candidate protocol and the gold protocol. It does *not* watch the video. So a model can be penalized for omitting a step the judge sees in the gold, even if the omission was correct (e.g., the researcher actually skipped that step).

## Where current models reportedly fail

Published baselines on Protocol Alignment cluster around 2.0–2.86 / 5. Failure modes the paper hints at:

- **Temporal grounding.** Knowing *when* a step starts and ends — frontier VLMs can describe what they see but blur step boundaries.
- **Parameter reading.** Reading a digital display on a vortexer, a handwritten label on a tube, a volume marker on a pipette tip. This is essentially scene-text recognition under bad lighting and motion blur.
- **Equipment recognition.** Distinguishing a 14 mL rack from a 50 mL rack, or a vortex from a thermomixer. The dedicated `labos1/segmentation` dataset exists precisely because foundation VLMs blow this.
- **Long-video coherence.** Once a video crosses 10 minutes, models start either compressing later steps or hallucinating cohesion that isn't there.

These are the failure modes our project's Phase 1 failure taxonomy is meant to map precisely — once we have a working harness, the next question is which of these is dragging the score down the most. That diagnosis is what should drive Phase 2 design choices, not a priori bets.
