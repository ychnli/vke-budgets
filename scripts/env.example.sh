# Template for scripts/env.sh, the local environment for vke-budgets.
# Copy it to scripts/env.sh (gitignored) and fill in your own paths:
#
#   cp scripts/env.example.sh scripts/env.sh
#
# Then `source scripts/env.sh` (never pipe it). The Slurm templates in
# scripts/ source it too.

# Activate a Python environment with this package installed
# (pip install -e .[test]), e.g. a conda env:
conda activate /path/to/your/env

# One thread per numerical library: the package runs windows in threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Optional, for the tests that need real data (they are skipped without):
#   VKE_TEST_MERRA2_ROOT  MERRA-2 root (one directory per year) holding
#                         2023-01-04 and 2023-01-05
#   VKE_TEST_REFERENCE    directory of the reference outputs for 2017-07-12
#                         (analysis_AR_IVKE1_{inst,tavg}20170712.nc)
# export VKE_TEST_MERRA2_ROOT=/path/to/merra2
# export VKE_TEST_REFERENCE=/path/to/reference/outputs
