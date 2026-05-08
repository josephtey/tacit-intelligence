#!/bin/bash
#SBATCH --job-name=lsv_gemini
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/slurm_logs/lsv_gemini_%j.out
#SBATCH --error=runs/slurm_logs/lsv_gemini_%j.err

# Gemini side of the eval: generate predictions for the 49 manifest records
# missing gemini-2.5-pro outputs, score them with Claude, then aggregate.
# Runs purely on CPU; this is API-bound work (Gemini + Anthropic).

set -euo pipefail

REPO="/home/joetey/tacit-intelligence"
PYTHON="/home/joetey/miniconda/bin/python"
cd "${REPO}"

echo "=== Gemini run+score ${SLURM_JOB_ID:-(local)} on $(hostname) ==="
echo "Started: $(date -Iseconds)"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at ${REPO}/.env" >&2
  exit 1
fi

echo
echo "=== [1/3] Generating gemini-2.5-pro ==="
"${PYTHON}" -m eval.run --models gemini-2.5-pro

echo
echo "=== [2/3] Scoring gemini-2.5-pro ==="
"${PYTHON}" -m eval.score --judge claude-opus-4-7 --models gemini-2.5-pro

echo
echo "=== [3/3] Aggregating report ==="
"${PYTHON}" -m eval.report --judge claude-opus-4-7

echo
echo "=== Done at $(date -Iseconds) ==="
echo "Report: ${REPO}/runs/report.md"
