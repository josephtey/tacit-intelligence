#!/bin/bash
#SBATCH --job-name=lsv_eval_report
#SBATCH --partition=hsu_gpu_priority
#SBATCH --account=hsulab
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm_logs/eval_report_%j.out
#SBATCH --error=runs/slurm_logs/eval_report_%j.err

# Aggregate report. Submitted with --dependency=afterok:<arrayjobid> so it
# runs after all 4 array tasks have completed successfully.

set -euo pipefail

REPO="/home/joetey/tacit-intelligence"
PYTHON="/home/joetey/miniconda/bin/python"
cd "${REPO}"

echo "=== Aggregate report ${SLURM_JOB_ID:-(local)} on $(hostname) ==="
echo "Started: $(date -Iseconds)"
echo

"${PYTHON}" -m eval.report --judge claude-opus-4-7

echo
echo "=== Done at $(date -Iseconds) ==="
echo "Report: ${REPO}/runs/report.md"
