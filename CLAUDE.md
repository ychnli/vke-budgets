# vke-budgets: context for agents

## Goal

A lightweight, readable Python package for computing and interpreting the vapor
kinetic energy (VKE) budgets of atmospheric rivers. Atmospheric scientists
should be able to plug in their own data. MERRA-2 comes first; ERA5 and GCM
output come later, so keep data-source specifics out of the physics.

Current phase: the core budget code. Priorities, in order:
1. Correctness: check for mistakes, both in the reference code and in ours.
2. The user's understanding of the calculation.
3. Clean code.

AR detection (TempestExtremes), box budgets and plotting are in scope but come
later. The end goal is to reproduce the main results of Zhang et al. 2026 from
the full MERRA-2 set.

The user (yucli) is new to this code and did not write the original NCL. Explain
the physics as you go, in docstrings (with the equation each function implements)
and in replies. Readability matters more than cleverness.

Repo: `$GROUP_HOME/yucli/vke-budgets`, remote `git@github.com:ychnli/vke-budgets.git`.

## References

The PDFs are in `papers/` (gitignored); read those rather than searching the
web.
- Ong & Yang 2024, *Nat. Commun.*, doi:10.1038/s41467-024-53369-0. Introduces
  VKE (q²K) for AR detection.
- Zhang, Yang, Ong & Tan 2026, *GRL*, doi:10.1029/2025GL119841. Introduces the
  KEV (qK) budget framework for AR evolution. "Zhang et al. 2026" in the code
  means this paper.

## Reference implementation (`$REF` = `$GROUP_SCRATCH/Aidi/VKE_budget`)

All existing code lives there, outside this repo. It is the ground truth for
numerical consistency: check every port of a calculation against it.

`reference/` in this repo is a read-only, gitignored snapshot of the code and
small files of `$REF` (taken 2026-10-15, with modification times kept), since
`$REF` is on purged scratch and has no VCS. `reference/MANIFEST.md` lists what
was left out and the sha256 of each file. Read code from `reference/`, which is
safe from the purge. Data and reference outputs are only in `$REF`.

- `$REF/case2023/py/ARanalysisMERRA2_IVKE1_VTE.py`: the KEV + VTE budget
  (`--vte`). This is the script being refactored. Its docstring, and that of
  `$REF/py/ARanalysisMERRA2_IVKE1.py` (the KEV-only original, identical
  without `--vte`), define every term and every departure from the NCL.
- `$REF/py/tools_comp.py` (`integZ`, `gradient_2d`, `divergence_2d`, via metpy)
  and `$REF/py/tools_ncl_equivalent.py` (ports of NCL `center_finite_diff_n`,
  `linint1_n` and `linmsg_n`): the numerical operators.
- `$REF/ncl/ARanalysisMERRA2_IVKE1.ncl`: the original NCL (Zhang).
  `$REF/case2023/ncl/ARanalysis_verify.ncl` is NCL with switches
  `integ`/`hgrad`/`order`. With trapz/fd/isobaric it matches the Python to
  float32 rounding.
- `$REF/case2023/external/AR_IVKE/`: Ong & Yang's NCL, including an ERA5
  version. It is the same code base as Zhang's, so it is not an independent
  check.
- `$REF/case2023/VERIFICATION_MERRA2.md`: every verification result and known
  numerical issue. Read it before changing any numerics.
- Downstream code for later phases: `$REF/py/budget_box_IVKE1.py` (box
  budgets), `$REF/py/frame_budget_video*.py` and `$REF/py/tools_plotting.py`
  (plots), `$REF/case2023/run_port.sbatch` (the DetectBlobs commands, with
  thresholds in `$REF/threshold_v2.txt`), and
  `$REF/case2023/py/make_website_budget.py` (term groupings and residual).
- `$REF/case2023/py/inst_dynamics.py` was a temporary experiment: the offline
  dynamics from the instantaneous fields, to probe the budget residual. It is
  not part of the package, but the residual is (see Later goals).
- `$REF` is Aidi's tree. Read it, never edit it; `$REF/case2023/` is ours.

## Physics in one screen

Definitions. q is specific humidity, K = (u² + v²)/2, and ⟨·⟩ = (1/g)∫ · dp over
1000–200 hPa.

| package name | definition | reference name | old NCL name |
|---|---|---|---|
| `kev` / `vte` (instantaneous columns) | ⟨qK⟩ / ⟨q²K⟩ | `IKEV` / `IVTE` | `IVKE1` / `IVKE` |
| `ivt` | \|⟨q**u**⟩\| | `IVT` | `IVT` |

KEV = qK is the budgeted quantity (Zhang et al. 2026); VTE = q²K is Ong &
Yang's "VKE". The NCL names only survive in the NCL scripts.

**Package names (user's decision, 2026-10-15).** Every budget variable is
`<quantity>_<term>` (`kev_…`, `vte_…`), the term named after its budget group;
column integrals are the default, `_3d` marks the same term on levels.
`src/vke_budgets/names.py` is the single table of names, groups, long names
and the translation to the reference's names (used by the legacy layout);
`docs/names.md` is generated from it (`python -m vke_budgets.names`), and a
test checks that it is current. Use the package names everywhere except when
talking about the reference's code or files.

Budget of E = qⁿK: ∂E/∂t = qⁿ **u**·∂**u**/∂t + n qⁿ⁻¹K ∂q/∂t. Groups, in
order (`budget.groups`), with members and the reference's names:

- `tend`: `kev_tend` (IKEVTEND), the change of the instantaneous column over
  the 3 h window.
- `hflux`: `kev_hflux_ke` = −qⁿ**u**·∇K (HAKE) + `kev_hflux_vapor` =
  −K∇·(qⁿ**u**) (HCV); together −⟨∇·(E**u**)⟩.
- `vflux`: `kev_vflux_ke` = −qⁿω**u**·∂**u**/∂p (VAKE) + `kev_vflux_vapor` =
  −K∂(qⁿω)/∂p (VCV).
- `pe_conversion`: −qⁿ**u**·∇Φ (PEKE).
- `subgrid_momentum` (the reference's "fric"): `_moist`, `_turb`, `_gwd`
  (MOKE, TOKE, GOKE): every parametrized force on the wind.
- `net_vapor_source` (the reference's "cond"): `_moist`, `_turb`, `_chem`
  (POV, TOV, COV): every source and sink of vapor.
- `analysis` (AOKEV); `boundary` (SPTE) = (∂p_s/∂t)·E(p_bot)/g, the Leibniz
  term; `residual` = tend minus all of the above.

Diagnostics: `kev_tend_local` (I_KEVTEND, the integral of the local
tendency); `kev_mean` (IKEVAVG); the advective form `kev_hadv_vapor`,
`kev_vadv_vapor` (HAV, VAV); `kev_model_dynamics_momentum`/`_vapor` (DOKE,
DOV: MERRA-2's own DYN tendencies, a cross-check on hflux + vflux +
pe_conversion); `vte_hflux_vapor_ncl`/`vte_vflux_vapor_ncl` (HCV_vO, VCV_vO,
the NCL's commented-out −2qK∇·(q**u**) form); the flux-form decomposition
`kev_fluxform_*`, `kev_pe_*`, `kev_continuity_error`, `divergence_3d_error`.

For VTE the momentum weight is q² and the vapor weight 2qK; `vte_hflux_vapor`
= −K∇·(q²**u**) is the user's chosen form. In MERRA-2 the residual is about
30% of the tendency; it is about 3% when MERRA-2's DYN tendencies replace the
offline dynamics.

## Numerical choices of the reference (they keep the numbers as they are)

- Vertical integral: trapezoid from the lowest above-ground level to 200 hPa
  (`integZ`), not NCL `vibeta`. `vibeta` drops the 1000–987.5 hPa half-layer
  when p_s > 1000 hPa.
- Horizontal derivatives: metpy finite differences on the lat-lon grid, not
  spherical harmonics. They tolerate below-ground NaNs. Output is blanked
  poleward of 87°.
- Horizontal derivatives are taken on isobaric surfaces by default; the `_eta`
  terms use the NCL's model-level ordering. Vertical derivatives are taken on
  the 72 hybrid levels (`tavg3_3d_asm_Nv`, 3D `PL`) and then linearly
  interpolated to pressure levels.
- The reference's horizontal operators have three flaws, confirmed against
  exact solutions in `tests/test_reference_operator_flaws.py`: (1) metpy
  without a CRS multiplies d/dy by a/M on top of the true spacing (+0.67% at
  the equator, −0.30% at 80°); (2) the divergence metric term uses a sphere
  (−v tanφ/R_mean) while the derivatives use WGS84; (3) d/dx is one-sided at
  the ±180° seam (twice the usual truncation error, at two columns).

## Our operators (`src/vke_budgets/{geometry,horizontal,vertical}.py`)

- Written in numpy, not metpy (agreed with the user: small dependencies,
  visible stencils). metpy is a test-only dependency.
- **The default is the correct calculation, with the flaws fixed (user's
  decision).** `HorizontalGeometry` has one switch per flaw, and the
  `geometry.REFERENCE` preset turns on all three plus WGS84. With it, the
  operators match the reference's to 1e-10 (`tests/test_reference_compat.py`).
  Use it to reproduce the reference outputs, and switch one flaw at a time
  to measure each fix.
- Default Earth: a sphere of radius 6371.0 km (`MERRA2_SPHERE`, GEOS's
  MAPL_RADIUS), agreed with the user: the model solves its equations on a
  sphere, and the original NCL assumes one (6.37122e6 m). WGS84 came in
  unintentionally through metpy; it stays available for `REFERENCE`.
- MAPL_GRAV is 9.80665 in MAPL v2.0.0 and later; the reference uses 9.81.
- The vertical operators follow the reference exactly: trapezoid integral
  skipping layers with a missing end, NCL-style two-layer d/dp, linear
  interpolation in p with no extrapolation.

## Notebooks

- The user keeps their own notebooks in `notebooks/` (commit them without
  outputs). The kernel `vke-budgets` (`~/.local/share/jupyter/kernels/vke`)
  sets `VKE_MERRA2_ROOT`, `VKE_CASE_OUT`, `VKE_AR_MASKS`,
  `VKE_MERRA2_ROOT_2017`, `VKE_JULY_OUT`, `VKE_AIDI_OUTPUT` and one thread per
  library; notebooks read paths only from those.

## Standard input and adapters

- `src/vke_budgets/standard.py` defines the standard input Dataset (names,
  dims, units) and `check_input`. Dims: `time` (window centres), `time_inst`
  (one more; window i runs from time_inst[i] to time_inst[i+1]), `lev`
  (pressure levels, Pa), `mlev` (model levels, bottom-up), `lat`, `lon`.
  Instantaneous `q_inst u_inst v_inst ps_inst` on `lev`; window means
  `p q u v phi omega` on `mlev`; optional physics tendencies
  `du/dv/dqdt_<process>` on `lev`.
- `src/vke_budgets/sources/merra2.py`: `find_files`, `open_day`. The result
  is lazy (dask, one chunk per time step); one window is 0.66 GB and reads in
  about 9 s. phi = g·H with the reference's g = 9.81 (a parameter).
- ERA5 will need the mean state on `lev` instead of `mlev`; not supported yet.

## Budget code

- `terms.py`: one function per term, written once for E = qⁿK (n = 1 KEV,
  n = 2 VTE) with weights w_m = qⁿ (momentum side) and w_v = n qⁿ⁻¹K (vapor
  side). hflux_vapor = −K∇·(qⁿ**u**) keeps hflux_ke + hflux_vapor = −∇·(E**u**)
  for both n.
- `budget.py`: `BudgetOptions` (geometry, quantities, g, limits, pole
  blanking at 87°, `keep_3d`, `dyn_crosscheck`, `ncl_vte_form`),
  `window_budget` (one window, pure), and `compute(ds, options, windows,
  n_workers)`, which returns window terms on `time` and IVT/IKEV/IVTE on
  `time_inst`.
- With `geometry.REFERENCE`, 2017-07-12 reproduces the `--no-3d` reference
  to ≤ 3e-7 relative RMS for every variable, with identical NaN footprints.
  IPEKE differs by 2.6e-5: the reference rounds Φ to float32 twice (in the
  interpolation and in ×9.81), and with that rounding emulated it matches to
  4e-8. The adapter now forms Φ in float64.
- 2023-01-04 (against the full `--vte` output) reproduces it the same way.
- `budget.groups(result, quantity)`: tend, hflux, vflux, pe_conversion,
  subgrid_momentum, net_vapor_source, analysis, boundary, residual (the
  reference's tend … resid of `make_website_budget.py`, renamed).
- Size of the deliberate operator changes, 2023-01-04, relative RMS of
  (default − reference): horizontal terms 0.6–1.0% away from the seam
  (IPEKE 1.5–1.8%, a near-geostrophic cancellation); 15–22% in the four
  columns at ±180° (the seam fix only touches those); exactly 0 for every
  term without a horizontal derivative. Closure in the AR masks improves
  slightly: KEV 0.283 → 0.281, VTE 0.340 → 0.334 (0.023 and 0.030 with
  MERRA-2's DYN tendencies). So the operator flaws are not the residual.
- Performance on 4 CPUs: 93 s per day with `n_workers=3`, 9.9 GB peak.
- `scripts/compare_reference.py` compares a day with the reference outputs,
  for `reference` and for one fix at a time.
- **Boundary term (option A, user's decision 2026-10-19).** `kev_boundary`
  (SPTE) is zero where p_s stays above p_bottom (1000 hPa) at both ends of the
  window: there the integrated column's bottom is fixed, so the Leibniz term
  is zero. `boundary_everywhere=True` (CLI `--boundary-everywhere`) restores
  the reference. Size: the term changes by 62–70% relative RMS (56% of the
  points set to zero); nothing else changes; closure is unchanged (KEV AR
  mask 0.2811 → 0.2815, Aidi's box L1/L1 0.3445 → 0.3440).
- `flux_form=True` adds the flux-form decomposition (`HFOKE`…`QPHIDIVV`,
  KEV only). Not yet ported: the `_eta` ordering.
- `budget.iter_windows` yields one window at a time (for streaming
  reductions); `compute` and `combine` build on it.

## AR detection and composites (`detection.py`, `composite.py`)

- `detection.detect(columns)`: wraps TempestExtremes DetectBlobs exactly as
  the reference (Laplacian{9,7.0} < threshold, ≥ 50 points, |lat| ≥ 20°;
  `REFERENCE_THRESHOLDS` from threshold_v2.txt). Masks match the reference's
  point for point (2023-01-03…05, all three fields). `window_mask`: an AR at
  either end of a window (the reference's mean ≥ 0.5).
- `composite.py`: `region`, `area_integral` (cos-lat), `ar_index` (1° box
  mean), `composite` (regression on the index times one standard deviation,
  with `lag` and `when`, as Ong & Yang 2024 and Zhang et al. 2026), and
  `contributions` (Zhang et al. Eqs. 8–10: growth and movement per term).
  Significance tests (t-test with lag autocorrelation) are not implemented.
  The paper's Text S2 (supplement) is not in `papers/`.

## Output and long runs (`io.py`, `pipeline.py`, `cli.py`)

- Clean layout: one file per calendar month, `budget_<first>-<last>.nc`,
  window terms on `time`, ivt/kev/vte on `time_inst`. A file excludes the
  instantaneous step that closes its last window, so files never overlap and
  `io.open_budget("budget_*.nc")` opens any number of them as one lazy Dataset.
- `open_budget` joins files with `budget.combine`, not `open_mfdataset`'s
  automatic combine: with two time dimensions, that kept the first file's
  window terms and filled the rest with NaN (a bug found by the notebook,
  fixed 2026-10-15; tests now compare values across files).
- Legacy layout: `write_legacy` / `open_legacy`, the reference's two files per
  day, under the reference's names (`names.to_legacy` / `from_legacy`; 1e15
  fill, integer minutes, levels in hPa). `open_legacy` also reads
  the reference's own outputs.
- `vke-budgets run --start --end --root --out [--layout legacy] [--bits N]`:
  resumable (finished days are kept as `parts/<day>.nc` until their month
  is merged; finished months are skipped). `scripts/run_merra2.sbatch`: one
  array task per month (~45 min on 4 CPUs); pass `--account`/`--partition`
  to `sbatch`.
- Sizes, one day of KEV + VTE (39 column variables, global): 200 MB lossless
  float32 + zlib; 121 MB at 12 BitRound bits (relative error 5e-5). Note
  that netCDF's `significant_digits` counts bits for BitRound. Raw MERRA-2
  input is 5.3 GB per day (Nv 2.2, inst 1.2, udt 1.15, qdt 0.73).

## MERRA-2 input specifics (put these in a MERRA-2 adapter, not in the physics)

- Five files per day: `tavg3_3d_qdt_Np`, `tavg3_3d_udt_Np` (physics tendencies),
  `tavg3_3d_asm_Nv` (model levels, top-down, reversed on read), and
  `inst3_3d_asm_Np` for the day **and the next day** (the tendency needs
  00Z of the next day).
- Time grids: instantaneous at 00, 03, …, 21Z; tavg windows centred at 01:30,
  04:30, …. Two consecutive instantaneous steps bracket each tavg window.
- `H` is geopotential height (Φ = gH). Fill value is 1e15. Below-ground points
  on Np are missing.
- Raw data on disk: 2017-07-12/13 in `$REF/data/2017`, and 2023-01-03…06 in
  `$REF/case2023/data/2023`.
- ERA5 differs: `z` is already geopotential, latitude is descending, there are
  no physics or analysis tendencies on pressure levels, and Ong & Yang's ERA5
  script computes only the dynamics terms. So the physics and analysis terms
  must be optional, and the residual absorbs them.

## Design decisions (agreed with the user)

- Outputs are xarray Datasets with clean names. The core returns a Dataset;
  writing to disk is a separate step.
- Keep backwards compatibility with the old written files
  (`analysis_AR_IVKE1_{inst,tavg}<date>.nc`). The package must be able to read
  them, and to write that layout, so existing outputs and downstream scripts
  stay usable.
- By default the package computes the core KEV and VTE budgets only. The
  diagnostic extras are opt-in: `_eta` ordering, flux-form decomposition
  (`HFOKE`…`QPHIDIVV`), the `_vO` pair, and the `DOKE`/`DOV` cross-checks.
- 3D (per-level) output fields are opt-in. By default only column integrals
  and the instantaneous IKEV/IVT/IVTE are returned and written.
- Every term is an xarray DataArray with coordinates, `units` and `long_name`,
  so users can inspect and plot terms directly.
- Separate the layers. Data-source adapters (names, units, level order,
  coordinates) produce a standard Dataset. Operators (integrate, grad, div,
  d/dp, interpolate) and budget-term functions are pure and source-agnostic.
- One function per physical term or small group of terms, named for the
  physics, with the equation and units in the docstring. Avoid the long
  monolithic `main()`.

## Numerical consistency (required)

- With `geometry.REFERENCE` and `boundary_everywhere=True`, the
  package must reproduce the reference outputs to float32 rounding. The
  defaults differ deliberately. Report every deliberate change to the user
  with its size (relative RMS per term, one change at a time).
- Compare with `$REF/case2023/verify/compare.py A.nc B.nc`, or an equivalent
  test in this repo. Reference outputs:
  - `$REF/case2023/output/analysis_AR_IVKE1_*2023010[345].nc`: full 3D, `--vte`
  - `$REF/case2023/verify/port_20170712/`: `--no-3d`
- Those reference outputs sit on purged scratch (90 days without
  modification). If they are gone, regenerate them with
  `$REF/case2023/run_port.sbatch`; don't refresh them to dodge the purge.
- Add fast unit tests on analytic fields (constant-integrand integrals,
  divergence of solid-body rotation, …). They need no MERRA-2 data.
  Full-day comparisons run as Slurm jobs.
- If the reference looks wrong, say so and quantify it. Don't silently
  replicate it, and don't silently fix it.

## Environment and cluster rules (Sherlock)

- **No site-specific paths or settings in tracked files** (`src/`, `scripts/`,
  `tests/`, README, pyproject); `tests/test_portability.py` enforces it. Site
  details go in the local, gitignored `scripts/env.sh` (copied from the
  tracked `scripts/env.example.sh`) or in environment variables. Ours
  activates the vke env and sets `VKE_TEST_MERRA2_ROOT` and
  `VKE_TEST_REFERENCE`, the data for the real-data tests (skipped without).
  `source scripts/env.sh` is enough for the package and the tests; NCL still
  needs `$REF/case2023/env_vke.sh`. This file (CLAUDE.md) is agent context
  and may hold site details.
- `source $REF/case2023/env_vke.sh`: a conda env at `$GROUP_HOME/yucli/envs/vke`
  (Python 3.12, numpy 2.0, xarray 2025.1, metpy 1.6.3) plus NCL 6.6.2. Source
  it; never pipe it. Run NCL with `< /dev/null`. Install this package into
  that env editable (`pip install -e .`).
- Full-day runs are Slurm jobs: `--account=dayang --partition=serc`, an explicit
  `--time`, and `--mem=40GB`. Template: `$REF/case2023/run_port.sbatch`.
  Never run them on a login node. Job outputs go to `$GROUP_SCRATCH` or
  `$SCRATCH`, never into this repo.
- Memory: the reference loops one tavg step at a time and still peaks near
  29 GB (40 GB with `--vte`) on the global 0.5° grid. Keep a per-time-step
  (or dask-chunked) design.

## Before the public release (agreed with the user)

- Remove CLAUDE.md from the public repository. It holds site-specific
  paths and settings on purpose; it is agent context for this cluster.
- Hardcoded paths that are already in the git history may stay there.

## Later goals

- Budget closure. The KEV residual is about 30% of the tendency with the
  offline dynamics and about 3% with MERRA-2's DYN tendencies
  (`VERIFICATION_MERRA2.md` §6). **Accepted (user and Aidi, 2026-10-18):** a
  larger residual from offline dynamics computed from 3-hourly output is
  largely unavoidable if the budget is to be resolved term by term; it is not
  treated as a bug. Report it with a fair normalization (L1/L1 or RMS
  pointwise; box residual against the size of the box budget), not against
  the signed, near-cancelling tendency. Parts that are not a sampling limit
  may still be worth fixing if the user asks: the terrain hot spots
  (isobaric derivatives next to the ground). The user prefers to take
  horizontal derivatives as the NCL did (2026-10-19); to be settled: the
  model-level ordering (`_eta`, portable) and/or spherical harmonics.
  Numbers so far (default geometry):
  - 2023-01-03…05, inside the IKEV AR mask: RMS(residual)/RMS(tend) 0.287;
    split exactly into tendency mismatch (tend − tend_local − boundary) 0.146,
    MERRA-2 closure 0.026, offline dynamics 0.281.
  - 2017-07-12, Aidi's box (170–120°W, 25–55°N): his "Residual of IKEV
    budget" (budget_box_IVKE1.py: MERRA-2 DYN, no SPTE) is 19.3% of the
    signed box tendency and 0.9% of its L1 norm (his 20% and 1%). Both
    signed ratios mislead: the box integral cancels 95% of the tendency and
    76% of his residual (91% of the offline one). Treated alike, his residual
    is 3.9% (L1/L1) and 4.4% (RMS); the offline one 34% and 42%. The box
    residual against the sum of |box groups|: 1.5% and 4.9%. The offline
    residual is largest over steep terrain (Alaska, Kamchatka, the Coast
    Mountains, Hawaii).
- Measure each operator fix on real data (relative RMS per term).
- Issues found while reading the reference, not yet acted on (raise them with
  the user when porting the code they affect):
  - Near steep terrain, horizontal derivatives on pressure levels are missing
    next to below-ground points, so the integral of those terms starts higher
    than the integral of the tendency. A possible source of residual.
  - `ddp` uses the two-layer chord (first-order on uneven levels).
  - g = 9.81 is used for Φ = gH and the integrals; MAPL_GRAV is 9.80665
    (not checked for the 2014 GEOS-5.12.4 of MERRA-2).
