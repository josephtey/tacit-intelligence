#!/bin/bash
#SBATCH --job-name=lsv_eval
#SBATCH --partition=hsu_gpu_priority
#SBATCH --account=hsulab
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm_logs/eval_%A_%a.out
#SBATCH --error=runs/slurm_logs/eval_%A_%a.err

# 4-task array. Each task processes a shard of the 50-video manifest:
#   shard 0/4: records 0-12   (13 records)
#   shard 1/4: records 13-24  (12 records)
#   shard 2/4: records 25-37  (12 records)
#   shard 3/4: records 38-49  (12 records)
# Each task runs both generators (gpt-5.5 + gemini-2.5-pro) and the Claude judge
# on its assigned shard. Predictions and scores cache by slice_id, so reruns
# are idempotent. The aggregate report is a separate dependent job.

set -euo pipefail

REPO="/home/joetey/tacit-intelligence"
PYTHON="/home/joetey/miniconda/bin/python"
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-4}"

mkdir -p "${REPO}/runs/slurm_logs"
cd "${REPO}"

echo "=== Job ${SLURM_JOB_ID:-(local)} task ${SLURM_ARRAY_TASK_ID:-?}/${SLURM_ARRAY_TASK_COUNT:-?} on $(hostname) ==="
echo "Started: $(date -Iseconds)"
echo "Shard:   ${SHARD}"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at ${REPO}/.env" >&2
  exit 1
fi

echo
echo "=== [1/2] Generating (shard ${SHARD}) ==="
"${PYTHON}" -m eval.run --shard "${SHARD}" --models gpt-5.5 gemini-2.5-pro

echo
echo "=== [2/2] Scoring (shard ${SHARD}) ==="
"${PYTHON}" -m eval.score --shard "${SHARD}" --judge claude-opus-4-7 --models gpt-5.5 gemini-2.5-pro

echo
echo "=== Shard ${SHARD} complete at $(date -Iseconds) ==="
