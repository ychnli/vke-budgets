# vke-budgets

`vke-budgets` computes the vapor kinetic energy budgets of atmospheric rivers
(ARs) from reanalysis data. Two related quantities are budgeted:

- **KEV** $= qK$, the kinetic energy of vapor (Zhang et al. 2026), and
- **VTE** $= q^2 K$, the vapor transport energy that Ong & Yang (2024) call VKE,

where $q$ is the specific humidity and $K = (u^2 + v^2)/2$ is the kinetic
energy per unit mass. For each 3-hour window, the package computes every term of the
budget (advection, conversion of potential energy, physics and analysis
tendencies) as a column integral from 1000 to 200 hPa. It can also detect ARs
and build regression composites that show how each term contributes to the
growth, decay and movement of a typical AR.

MERRA-2 is supported now. Other data sources can be used by converting them to
the package's standard input format, described below.

## The budget

The budget of $E = q^n K$ ($n = 1$ for KEV, $n = 2$ for VTE) follows from the
momentum and moisture equations:

$$
\frac{\partial E}{\partial t} = q^n \, \mathbf{u} \cdot \frac{\partial \mathbf{u}}{\partial t}
+ n \, q^{n-1} K \, \frac{\partial q}{\partial t},
$$

where $\mathbf{u} = (u, v)$ is the horizontal wind. Every term is integrated over
the column,

$$
\langle X \rangle = \frac{1}{g} \int_{200\,\mathrm{hPa}}^{1000\,\mathrm{hPa}} X \, dp,
$$

where $g$ is the gravitational acceleration. Over high ground, the integral
starts at the lowest pressure level above the surface. The terms fall into
groups, so that

$$
\mathtt{tend} = \mathtt{hflux} + \mathtt{vflux} + \mathtt{pe\_conversion}
+ \mathtt{subgrid\_momentum} + \mathtt{net\_vapor\_source} + \mathtt{analysis}
+ \mathtt{boundary} + \mathtt{residual}.
$$

| group | meaning |
|---|---|
| `tend` | change of the column total over the 3-hour window |
| `hflux` | horizontal flux divergence of $E$, $-\langle \nabla \cdot (E \mathbf{u}) \rangle$ |
| `vflux` | vertical flux divergence of $E$, $-\langle \partial (E \omega) / \partial p \rangle$, with $\omega$ the vertical pressure velocity |
| `pe_conversion` | conversion of potential energy into kinetic energy, $-\langle q^n \, \mathbf{u} \cdot \nabla \Phi \rangle$, with $\Phi$ the geopotential |
| `subgrid_momentum` | parametrized forces on the wind: turbulence, convection and gravity-wave drag |
| `net_vapor_source` | all sources and sinks of vapor: condensation, evaporation, turbulent transport and chemistry |
| `analysis` | analysis increments of the reanalysis |
| `boundary` | the effect of a moving lower boundary (surface pressure tendency) |
| `residual` | what the other groups do not account for |

Each output variable is named `<quantity>_<term>`, with the term named after
its group: for example `kev_hflux_ke`, `kev_hflux_vapor` and
`vte_subgrid_momentum_turb`. Column integrals carry the plain name, and the
same term on pressure levels adds `_3d`. Every variable has `units`,
`long_name` and `budget_group` attributes. [`docs/names.md`](docs/names.md)
lists all of them, together with the names used by the original reference
code (IHAKE, SPTE and so on).

The residual is not small when the dynamical terms are computed from 3-hourly
reanalysis output: point by point it is typically 30 to 40% of the tendency. It
largely cancels in area integrals, and it is a few percent when MERRA-2's own
dynamics tendencies replace the offline ones. When reporting it, normalize the
residual and the tendency in the same way (for example RMS against RMS),
because the area integral of the tendency alone nearly cancels.

## Installation

The package needs Python 3.10 or newer. In a fresh environment (conda is
convenient because TempestExtremes is a conda package):

```bash
conda create -n vke -c conda-forge python=3.12 tempest-extremes
conda activate vke
git clone git@github.com:ychnli/vke-budgets.git
cd vke-budgets
pip install -e ".[test]"
```

TempestExtremes is only needed for AR detection. The package computes windows
in parallel threads, so set the numerical libraries to one thread each:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`scripts/env.example.sh` is a template for a local environment script that
does this. Copy it to `scripts/env.sh`, which git ignores, and fill in your
own paths.

## Input data

For MERRA-2, the budget of one day uses five files, kept in one directory per
year under a common root:

    <root>/2023/MERRA2_400.inst3_3d_asm_Np.20230104.nc4   instantaneous state
    <root>/2023/MERRA2_400.inst3_3d_asm_Np.20230105.nc4   (00 UTC of the next day is needed)
    <root>/2023/MERRA2_400.tavg3_3d_asm_Nv.20230104.nc4   3-hour means on model levels
    <root>/2023/MERRA2_400.tavg3_3d_udt_Np.20230104.nc4   wind tendencies by process
    <root>/2023/MERRA2_400.tavg3_3d_qdt_Np.20230104.nc4   vapor tendencies by process

These come to about 5.3 GB per day. `sources.merra2.open_day` reads them
lazily and returns the standard input Dataset, so only one 3-hour window is
held in memory at a time.

For another data source, write an adapter that produces the same standard
Dataset. `vke_budgets/standard.py` describes its variables, dimensions and
units, and `standard.check_input` reports anything that does not conform. The
physics tendencies are optional: without them, their groups are zero and the
residual absorbs them.

## Computing the budget of one day

```python
from vke_budgets import budget
from vke_budgets.sources import merra2

day = merra2.open_day("20230104", "/path/to/merra2")
result = budget.compute(day, n_workers=3)        # about 1.5 minutes on 4 CPUs
```

`result` is an xarray Dataset. The budget terms are on `time`, the centres of
the eight 3-hour windows. The instantaneous columns `ivt`, `kev` and `vte` are
on `time_inst`, the nine instants that bound those windows. To see the budget
by group:

```python
groups = budget.groups(result, "KEV")
groups["residual"].isel(time=4).plot()
```

To compute only some windows (for example for a case study), pass their
indices: `budget.compute(day, windows=[3, 4])`. The function
`budget.iter_windows` yields one window at a time, which is useful for
accumulating statistics without keeping every window in memory.

### Options

`budget.BudgetOptions` controls what is computed and how. The most useful
options are:

| option | default | effect |
|---|---|---|
| `quantities` | `("KEV", "VTE")` | which budgets to compute |
| `keep_3d` | `False` | also return every term on pressure levels (much larger output) |
| `dyn_crosscheck` | `False` | also compute the terms of the model's own dynamics tendencies, a check on the offline dynamics |
| `flux_form` | `False` | also compute the flux-form decomposition of the KEV budget |
| `ncl_vte_form` | `False` | also compute the alternative VTE convergence terms of the original NCL code |
| `boundary_everywhere` | `False` | apply the boundary term at every point, as the reference code does |
| `geometry` | `geometry.DEFAULT` | the shape of the Earth and the finite-difference choices |

For example:

```python
options = budget.BudgetOptions(quantities=("KEV",), keep_3d=True, dyn_crosscheck=True)
result = budget.compute(day, options, windows=[4])
```

## Long runs from the command line

For many days, use the `vke-budgets` command. It writes one netCDF file per
calendar month, and it can be interrupted and restarted: finished days are
kept until their month is complete, and finished months are skipped.

```bash
vke-budgets run --start 2023-01-01 --end 2023-12-31 \
    --root /path/to/merra2 --out /path/to/output --workers 3
```

`vke-budgets run --help` lists the options, which follow `BudgetOptions`. The
option `--bits 12` rounds the stored values to 12 significant bits (about
3.6 decimal digits), which shrinks a day of output from about 200 MB to
120 MB.

The month files open together as one Dataset, loaded lazily:

```python
from vke_budgets import io

ds = io.open_budget("/path/to/output/budget_*.nc")
```

`scripts/run_merra2.sbatch` is a Slurm template that processes one month per
array task.

## Detecting atmospheric rivers

`detection.detect` finds ARs as Ong & Yang (2024) do, with TempestExtremes. An
AR is a blob of at least 50 connected grid points, poleward of $20^\circ$, where the
Laplacian of a column field (IVT, KEV or VTE) is below a threshold. The
default thresholds are those of the reference code.

```python
from vke_budgets import detection

masks = detection.detect(ds[["ivt", "kev", "vte"]])      # mask_ivt, mask_kev, mask_vte
ar_in_window = detection.window_mask(masks["mask_kev"], ds["time"].values)
```

The masks are on the instantaneous times. `window_mask` converts a mask to the
3-hour windows of the budget terms, counting an AR at either end of a window.

## Composites and the contributions of each term

A regression composite describes a typical AR passing a chosen point
(Ong & Yang 2024; Zhang et al. 2026). The AR index $I_0(t)$ is the column KEV
averaged over a $1^\circ \times 1^\circ$ box around the point. At every grid
point, each field $X$ is regressed on the index, and the regression coefficient
is multiplied by one standard deviation $\sigma_0$ of the index:

$$
X_{\mathrm{composite}} = \beta \, \sigma_0,
\qquad
\beta = \frac{\sum_t X'(t) \, I_0'(t)}{\sum_t I_0'(t)^2},
$$

where the prime marks the departure from the time mean. The composite therefore
shows the field that comes with a one-standard-deviation AR at the point.

`composite.contributions` then measures, over a domain $S$ around the point,
how much each budget term $X$ contributes to the growth or decay of the
composite AR and to its movement (Zhang et al. 2026, Eqs. 8 to 10). With $I$
the composite column KEV and $\partial I / \partial t$ its composite tendency,

$$
G(X) = \frac{\iint_S X \, I \, dS}{\iint_S I^2 \, dS},
\qquad
M(X) = \frac{\iint_S X \, \tilde{I}_t \, dS}{\iint_S \tilde{I}_t^{\,2} \, dS},
\qquad
\tilde{I}_t = \frac{\partial I}{\partial t} - G\!\left(\frac{\partial I}{\partial t}\right) I .
$$

$G(X)$ is a growth rate (per second). $\tilde{I}_t$ is the tendency with its
uniform growth or decay removed, which is the part that moves the AR, and
$M(X)$ is the share of that movement due to $X$. Both are linear in $X$, so the
contributions of the terms add up to those of the total tendency.

```python
from vke_budgets import budget, composite

point = {"lon": 180.3, "lat": 37.75}                     # degrees east (0-360) and north
domain = {"lon": (150.3, 210.3), "lat": (22.75, 52.75)}

index = composite.ar_index(ds["kev_mean"], **point)
# True in the windows with an AR at the grid point nearest the composite point
at_point = ar_in_window.sel(lat=point["lat"], lon=point["lon"] - 360, method="nearest")
terms = budget.groups(ds, "KEV")

terms_composite = composite.composite(terms, index, when=at_point)
kev_composite = composite.composite(ds["kev_mean"], index, when=at_point)
shares = composite.contributions(terms_composite.drop_vars("tend"), kev_composite,
                                 terms_composite["tend"], **domain).compute()
print(shares["growth"] * 86400)                          # per day
print(shares["movement"])                                # shares that sum to 1
```

Use `lag` in `composite.composite` to follow the composite AR before and after
it passes the point; one step is one 3-hour window. Composites need many
years of output to be meaningful (Zhang et al. use 2010 to 2019).

## Compatibility with the reference code

The package started as a refactoring of an existing Python translation of NCL
scripts. It can still read and write that code's output files, two per day
(`analysis_AR_IVKE1_inst<date>.nc` and `analysis_AR_IVKE1_tavg<date>.nc`),
with the original variable names:

```python
io.write_legacy(result, "/path/to/output", "20230104")
old = io.open_legacy("/path/to/reference/output", "20230104")   # read with the package's names
```

`vke-budgets run --layout legacy` writes that layout directly.

The package's defaults differ from the reference code in a few deliberate
ways. Horizontal derivatives assume a spherical Earth and wrap around in
longitude, and the boundary term is zero where the bottom of the integrated
column cannot move. With `geometry=geometry.REFERENCE` and
`boundary_everywhere=True`, the package reproduces the reference outputs to
float32 rounding.

## Tests

```bash
python -m pytest
```

Most tests use analytic fields with known answers and need no data. The tests
that use real data are skipped unless these environment variables point to
it:

| variable | contents |
|---|---|
| `VKE_TEST_MERRA2_ROOT` | MERRA-2 files for 2023-01-04 and 2023-01-05 |
| `VKE_TEST_REFERENCE` | reference outputs for 2017-07-12 |
| `VKE_TEST_AR_MASKS` | reference AR masks for 2023-01-04 |

The tests against the reference code's own functions also need a copy of
that code in `reference/` and metpy, which `pip install -e ".[test]"`
installs.

## References

- Ong, H., & Yang, D. (2024). Vapor kinetic energy for the detection and
  understanding of atmospheric rivers. *Nature Communications*, 15, 9428.
  doi:10.1038/s41467-024-53369-0
- Zhang, A., Yang, D., Ong, H., & Tan, Z. (2026). Understanding the evolution
  of global atmospheric rivers with a vapor kinetic energy framework.
  *Geophysical Research Letters*, 53, e2025GL119841. doi:10.1029/2025GL119841
