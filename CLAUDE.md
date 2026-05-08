Project -- 

Science runs on tacit knowledge that never gets written down. A protocol lists the steps of an experiment; the
execution — the way a skilled researcher actually performs those steps — lives only in hands and eyes. When
experienced scientists leave, that execution knowledge leaves with them. The reproducibility crisis is the visible
symptom.
Making scientific execution legible to machines is the prerequisite for almost every downstream ambition in
AI-for-science: reliable automation, meaningful robot operators, AI collaborators that can follow along with what
a bench scientist is actually doing. But before any of that, a simpler question has to be answered: can a model
watch a lab video and faithfully reconstruct what happened?
Today, frontier vision-language models cannot. On LabSuperVision (Cong et al., 2025, arXiv:2510.14861) — a
recently released benchmark of ~200 egocentric lab videos — GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-7B all
score between 2.0 and 2.86 out of 5 on Protocol Alignment, the task of transcribing a video into a structured,
parameterized procedure. Only LabOS's in-house 235B-parameter VLM closes the gap.
This project takes Protocol Alignment on LabSuperVision as its North Star metric and tries to build a system —
whatever its form — that meaningfully moves the score. The form is deliberately left open. It could be a small
fine-tuned VLM. It could be a compound agent orchestrating frontier models with visual tool use, memory, and
structured decomposition. It could be a hybrid. Finding the right shape is the research question, not an
assumption.
--The work organizes around a single goal — build a Protocol Alignment system that meaningfully beats the
strongest zero-shot frontier baseline — and four questions that have to be answered along the way:
1. Where does the frontier actually break? A rigorous reproduction of LSV baselines and a structured failure
taxonomy across protocol types and error classes. What exactly is the gap made of? Every subsequent design
decision is driven by this diagnosis.
2. Train, compose, or hybrid? Explore the full spectrum from pure fine-tuning (LoRA on small open-weight VLMs)
to pure compound agents (frontier VLM + visual tool use + decomposition + memory) to hybrids (a small
fine-tuned perception model feeding a larger reasoning agent). The question is empirical — measure, against
the North Star, which shape of system actually works and why. This is the most important question for any
early-stage company deciding whether to train its own models or build on top of frontier ones.
3. What signal actually matters? Ablate systematically across modalities (video, audio narration, object
detections, equipment state) and data regimes (training data scale, context length, tool calls, reasoning depth).
Use the results to answer the question any real lab-observability system has to face: which signals are worth the
cost of capturing them?
4. What does utility look like beyond the rubric? Show outputs from the best system to 2–3 wet-lab scientists on
real procedures. Collect qualitative signal on where the rubric and real-researcher judgment diverge. A
benchmark cannot measure this dimension; only users can.