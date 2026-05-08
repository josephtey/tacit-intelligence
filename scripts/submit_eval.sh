#!/bin/bash
# Submit the LSV eval as a 4-task array, then queue the aggregate report
# as a dependent job. Print job IDs and tracking commands.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
mkdir -p runs/slurm_logs

ARRAY_OUT=$(sbatch scripts/eval_lsv.sbatch)
echo "${ARRAY_OUT}"
ARRAY_JOBID=$(echo "${ARRAY_OUT}" | awk '{print $NF}')

REPORT_OUT=$(sbatch --dependency=afterok:"${ARRAY_JOBID}" scripts/eval_lsv_report.sbatch)
echo "${REPORT_OUT}"
REPORT_JOBID=$(echo "${REPORT_OUT}" | awk '{print $NF}')

cat <<EOF

=================================================================
Submitted:
  Array (4 tasks):  ${ARRAY_JOBID}
  Report (deps):    ${REPORT_JOBID}

Track progress:
  squeue -u \$USER -j ${ARRAY_JOBID},${REPORT_JOBID}
  scripts/progress.sh
  tail -f runs/slurm_logs/eval_${ARRAY_JOBID}_*.out
  tail -f runs/slurm_logs/eval_report_${REPORT_JOBID}.out

Cancel:
  scancel ${ARRAY_JOBID} ${REPORT_JOBID}
=================================================================
EOF
