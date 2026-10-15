#!/bin/bash
#SBATCH --job-name=vke_budgets
#SBATCH --account=dayang
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --output=logs/%x.%A_%a.out

# KEV and VTE budgets for one month of MERRA-2 per array task (a month is
# about 45 min on 4 CPUs, with 3 windows in flight). Submit one array per
# year, from the repository root (logs/ must exist):
#
#   YEAR=2023 ROOT=/path/to/merra2 OUT=$SCRATCH/vke-budgets/clim sbatch --export=ALL scripts/run_merra2.sbatch
#
# ROOT holds one directory per year of raw files. The last day of a month
# needs 00Z of the next day's inst3_3d_asm_Np. Reruns skip finished months
# and resume unfinished ones from their last finished day. Set BITS=12 to
# round the month files to 12 significant bits (about 120 MB per day instead
# of 200).
set -euo pipefail
YEAR="${YEAR:?set YEAR}"
ROOT="${ROOT:?set ROOT}"
OUT="${OUT:?set OUT}"
MONTH=$(printf %02d "$SLURM_ARRAY_TASK_ID")
START="$YEAR-$MONTH-01"
END=$(date -d "$START +1 month -1 day" +%F)

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
mkdir -p "$OUT"
echo "host=$(hostname) start=$START end=$END root=$ROOT out=$OUT"
vke-budgets run --start "$START" --end "$END" --root "$ROOT" --out "$OUT" \
    --workers 3 ${BITS:+--bits "$BITS"}
