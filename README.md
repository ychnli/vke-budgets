# vke-budgets

`vke-budgets` computes the vapor kinetic energy budgets of atmospheric rivers
(ARs) from MERRA-2 reanalysis, and detects ARs. Two related quantities are
budgeted:

- **KEV** $= qK$, the kinetic energy of vapor (Zhang et al. 2026), and
- **VTE** $= q^2 K$, the vapor transport energy that Ong & Yang (2024) call VKE,

where $q$ is the specific humidity and $K = (u^2 + v^2)/2$ is the kinetic
energy per unit mass. For each 3-hour window, the package computes every term
of the budget as a column integral from 1000 to 200 hPa. The terms are grouped
so that

$$
\mathtt{tend} = \mathtt{hflux} + \mathtt{vflux} + \mathtt{pe\_conversion}
+ \mathtt{subgrid\_momentum} + \mathtt{net\_vapor\_source} + \mathtt{analysis}
+ \mathtt{boundary} + \mathtt{residual},
$$

where `tend` is the change of the column total over the window, `hflux` and
`vflux` are the horizontal and vertical flux divergences, `pe_conversion` is
the conversion of potential energy into kinetic energy, `subgrid_momentum` is
the parametrized forcing of the wind, `net_vapor_source` is the net source of
vapor, `analysis` is the reanalysis increments and `boundary` is the effect of
the changing surface pressure. [`docs/names.md`](docs/names.md) lists every
output variable.

## Installation

The package needs Python 3.10 or newer. AR detection also needs
TempestExtremes, which is a conda package:

```bash
conda create -n vke -c conda-forge python=3.12 tempest-extremes
conda activate vke
git clone git@github.com:ychnli/vke-budgets.git
cd vke-budgets
pip install -e .
```

The package computes windows in parallel threads, so set the numerical
libraries to one thread each:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Input data

The budget of one day uses five MERRA-2 files, kept in one directory per year
under a common root:

    <root>/2023/MERRA2_400.inst3_3d_asm_Np.20230104.nc4   instantaneous state
    <root>/2023/MERRA2_400.inst3_3d_asm_Np.20230105.nc4   (00 UTC of the next day is needed)
    <root>/2023/MERRA2_400.tavg3_3d_asm_Nv.20230104.nc4   3-hour means on model levels
    <root>/2023/MERRA2_400.tavg3_3d_udt_Np.20230104.nc4   wind tendencies by process
    <root>/2023/MERRA2_400.tavg3_3d_qdt_Np.20230104.nc4   vapor tendencies by process

These files come to about 5.3 GB per day.

## Computing the budget of one day

```python
from vke_budgets import budget
from vke_budgets.sources import merra2

day = merra2.open_day("20230104", "/path/to/merra2")
result = budget.compute(day, n_workers=3)        # about 1.5 minutes on 4 CPUs
groups = budget.groups(result, "KEV")            # the KEV budget by group
groups["pe_conversion"].isel(time=4).plot()
```

`result` is an xarray Dataset. The budget terms are on `time`, the centres of
the eight 3-hour windows of the day, and are named `<quantity>_<term>`, for
example `kev_hflux_ke` or `vte_pe_conversion`. The instantaneous column fields
`ivt`, `kev` and `vte` are on `time_inst`, the nine instants that bound those
windows.

## Computing the budget of many days

The `vke-budgets` command computes a range of days and writes one netCDF file
per calendar month. If it is interrupted, running it again continues where it
stopped.

```bash
vke-budgets run --start 2023-01-01 --end 2023-12-31 \
    --root /path/to/merra2 --out /path/to/output --workers 3
```

The month files open together as one Dataset:

```python
from vke_budgets import io

ds = io.open_budget("/path/to/output/budget_*.nc")
```

## Detecting atmospheric rivers

`detection.detect` finds ARs with TempestExtremes, as Ong & Yang (2024) do. An
AR is a blob of at least 50 connected grid points, poleward of $20^\circ$,
where the Laplacian of a column field is below a threshold. The field can be
the column KEV, the column VTE or IVT.

```python
from vke_budgets import detection

masks = detection.detect(ds[["kev", "vte"]])     # mask_kev and mask_vte
masks["mask_kev"].isel(time_inst=0).plot()
```

The masks are on the instantaneous times. To select the 3-hour windows of the
budget terms that have an AR at either end, use
`detection.window_mask(masks["mask_kev"], ds["time"].values)`.

## AR frequency

The `vke-budgets climatology` command computes IVT and the column KEV and
VTE at every 3-hourly instant, detects ARs in each field, and writes one file
per calendar month. The file holds, for each field and grid point, the number
of time steps with an AR (`ar_count_kev`, for example), the AR frequency (the
fraction of time steps with an AR), and the monthly mean of the field. Only
the instantaneous files of the input are read.

```bash
vke-budgets climatology --start 2010-01 --end 2019-12 \
    --root /path/to/merra2 --out /path/to/output
```

The AR frequency over many months is the sum of `ar_count_<field>` over the
month files divided by the sum of `n_steps`. Multiplied by 365.25, it is in
days per year. A month takes about 8 minutes on 2 CPUs. For array jobs,
`--task` and `--months-per-task` select one group of months.

## References

- Ong, H., & Yang, D. (2024). Vapor kinetic energy for the detection and
  understanding of atmospheric rivers. *Nature Communications*, 15, 9428.
  doi:10.1038/s41467-024-53369-0
- Zhang, A., Yang, D., Ong, H., & Tan, Z. (2026). Understanding the evolution
  of global atmospheric rivers with a vapor kinetic energy framework.
  *Geophysical Research Letters*, 53, e2025GL119841. doi:10.1029/2025GL119841
