#!/bin/bash
#SBATCH --job-name=merra2_download
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x.%A_%a.out

# One month of MERRA-2 per array task (scripts/download_merra2.py), task 0
# being the month FIRST (YYYY-MM). A month fetches about 165 GB and keeps
# about 82 GB (without the DYN tendencies); it takes about 1.7 h, mostly
# nccopy cutting the files. Submit from the repository root (logs/ must
# exist), giving your cluster's account and partition on the command line,
# e.g. 2010-2019, 8 months at a time:
#
#   FIRST=2010-01 ROOT=/path/to/merra2 \
#       sbatch --array=0-119%8 --account=<account> --partition=<partition> \
#       --export=ALL scripts/download_merra2.sbatch
#
# The %8 caps the number of tasks downloading at once. Each task also fetches
# the inst3_3d_asm_Np file of the 1st of the next month. Reruns skip the
# files already in ROOT. The whole files wait in VKE_DOWNLOAD_WORK while they
# are cut (one at a time, up to 2.3 GB): node-local disk is best, set it in
# scripts/env.sh; the default is ROOT/.work.
set -euo pipefail
FIRST="${FIRST:?set FIRST (YYYY-MM)}"
ROOT="${ROOT:?set ROOT}"
START=$(date -d "$FIRST-01 +$SLURM_ARRAY_TASK_ID month" +%F)
END=$(date -d "$START +1 month -1 day" +%F)

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
WORK="${VKE_DOWNLOAD_WORK:-$ROOT/.work}"
echo "host=$(hostname) start=$START end=$END root=$ROOT work=$WORK"
python "$SLURM_SUBMIT_DIR/scripts/download_merra2.py" --start "$START" --end "$END" \
    --root "$ROOT" --work "$WORK"
