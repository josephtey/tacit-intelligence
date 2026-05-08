#!/bin/bash
#SBATCH --job-name=lsv_eval_score
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=runs/slurm_logs/eval_score_%j.out
#SBATCH --error=runs/slurm_logs/eval_score_%j.err

# Score the 50 cached gpt-5.5 predictions with Claude Opus 4.7, then
# aggregate the report. CPU-only — Claude API calls + JSON parsing.

set -euo pipefail

REPO="/home/joetey/tacit-intelligence"
PYTHON="/home/joetey/miniconda/bin/python"
cd "${REPO}"

echo "=== Score job ${SLURM_JOB_ID:-(local)} on $(hostname) ==="
echo "Started: $(date -Iseconds)"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at ${REPO}/.env" >&2
  exit 1
fi

echo
echo "=== [1/2] Scoring gpt-5.5 with Claude Opus 4.7 ==="
"${PYTHON}" -m eval.score --judge claude-opus-4-7 --models gpt-5.5

echo
echo "=== [2/2] Aggregating report ==="
"${PYTHON}" -m eval.report --judge claude-opus-4-7

echo
echo "=== Done at $(date -Iseconds) ==="
echo "Report: ${REPO}/runs/report.md"
