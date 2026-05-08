#!/bin/bash
# Live status snapshot for the LSV eval.
# Run anytime: scripts/progress.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

MANIFEST_N=$(/home/joetey/miniconda/bin/python -c "import json; print(len(json.load(open('runs/manifests/bench_50_seed42.json'))))")

count() { [[ -d "$1" ]] && find "$1" -maxdepth 1 -name '*.json' | wc -l || echo 0; }

PRED_GPT=$(count "runs/predictions/gpt-5.5")
PRED_GEM=$(count "runs/predictions/gemini-2.5-pro")
SCORE_GPT=$(count "runs/scores/claude-opus-4-7/gpt-5.5")
SCORE_GEM=$(count "runs/scores/claude-opus-4-7/gemini-2.5-pro")

printf "\n=== LSV eval progress (manifest=%d) ===\n" "${MANIFEST_N}"
printf "  Generation:  gpt-5.5         %3d/%d\n" "${PRED_GPT}" "${MANIFEST_N}"
printf "  Generation:  gemini-2.5-pro  %3d/%d\n" "${PRED_GEM}" "${MANIFEST_N}"
printf "  Scoring:     claude<-gpt-5.5         %3d/%d\n" "${SCORE_GPT}" "${MANIFEST_N}"
printf "  Scoring:     claude<-gemini-2.5-pro  %3d/%d\n" "${SCORE_GEM}" "${MANIFEST_N}"

echo
echo "=== SLURM queue ==="
squeue -u "${USER}" --noheader -o "  %i  %j  %T  %M  %R" || true

echo
echo "=== Latest log lines ==="
shopt -s nullglob
LOGS=(runs/slurm_logs/eval_*.out)
if (( ${#LOGS[@]} == 0 )); then
  echo "  (no logs yet)"
else
  for log in "${LOGS[@]}"; do
    printf "\n--- %s ---\n" "$(basename "${log}")"
    tail -n 3 "${log}"
  done
fi
echo
