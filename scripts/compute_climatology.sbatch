#!/bin/bash
#SBATCH --job-name=ar_climatology
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x.%A_%a.out

# AR statistics for MONTHS_PER_TASK consecutive months (default 3) per array
# task (scripts/compute_climatology.py), task i doing the i-th group of the
# months from START to END (YYYY-MM). A month takes about 8 min (about 13 s
# per day for the columns, then one DetectBlobs run per field). Memory peaks
# near 6 GB during detection (about 3 GB here and 3 GB in DetectBlobs). Submit from
# the repository root (logs/ must exist), giving your cluster's account and
# partition on the command line, e.g. 2010-2019 (120 months, 40 tasks):
#
#   START=2010-01 END=2019-12 ROOT=/path/to/merra2 OUT=/path/to/output \
#       sbatch --array=0-39 --account=<account> --partition=<partition> \
#       --export=ALL scripts/compute_climatology.sbatch
#
# Set SAVE_STEPS=1 to also write the 3-hourly columns and masks (about
# 0.5 GB per month). Reruns skip the months already written.
set -euo pipefail
START="${START:?set START (YYYY-MM)}"
END="${END:?set END (YYYY-MM)}"
ROOT="${ROOT:?set ROOT}"
OUT="${OUT:?set OUT}"
MONTHS_PER_TASK="${MONTHS_PER_TASK:-3}"

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID range=$START..$END root=$ROOT out=$OUT"
python "$SLURM_SUBMIT_DIR/scripts/compute_climatology.py" --start "$START" --end "$END" \
    --root "$ROOT" --out "$OUT" --task "$SLURM_ARRAY_TASK_ID" \
    --months-per-task "$MONTHS_PER_TASK" ${SAVE_STEPS:+--save-steps}
