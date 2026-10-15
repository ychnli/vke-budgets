# Environment for vke-budgets (source it, never pipe it): the conda env at
# $GROUP_HOME/yucli/envs/vke, with this package installed editable.
# One thread per numerical library: the package runs windows in threads.
eval "$(conda shell.bash hook 2>/dev/null)" || source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate "$GROUP_HOME/yucli/envs/vke"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
