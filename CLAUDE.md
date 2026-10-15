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

Paper PDFs go in this repo once the user adds them; read those rather than
searching the web.
- Ong & Yang 2024, *Nat. Commun.*, doi:10.1038/s41467-024-53369-0. Introduces
  VKE (q²K) for AR detection.
- Zhang, Yang, Ong & Tan 2026, *GRL*, doi:10.1029/2025GL119841. Introduces the
  KEV (qK) budget framework for AR evolution. "Zhang et al. 2026" in the code
  means this paper.

## Reference implementation (`$REF` = `$GROUP_SCRATCH/Aidi/VKE_budget`)

All existing code lives there, outside this repo. It is the ground truth for
numerical consistency: check every port of a calculation against it.

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
- `$REF` is Aidi's tree. Read it, never edit it; `$REF/case2023/` is ours.

## Physics in one screen

Definitions. q is specific humidity, K = (u² + v²)/2, and ⟨·⟩ = (1/g)∫ · dp over
1000–200 hPa.

| current name | definition | old NCL name |
|---|---|---|
| `KEV` / `IKEV` | qK / ⟨qK⟩, the budgeted quantity | `VKE1` / `IVKE1` |
| `VTE` / `IVTE` | q²K / ⟨q²K⟩, Ong & Yang's VKE | `VKE` / `IVKE` |
| `IVT` | \|⟨q**u**⟩\| | `IVT` |

Always use the current names. The NCL names only survive in the NCL scripts.

KEV budget: ∂(qK)/∂t = q **u**·∂**u**/∂t + K ∂q/∂t. Substituting the momentum and
moisture equations gives one term per process. Momentum-side terms end in `KE`
(weight q**u**·); vapor-side terms end in `V` (weight K):

- `HAKE` = −q **u**·∇K, `VAKE` = −q ω **u**·∂**u**/∂p, `PEKE` = −q **u**·∇Φ (PE→KE conversion)
- `HAV` = −K **u**·∇q, `VAV` = −K ω ∂q/∂p. Flux-form alternatives:
  `HCV` = −K ∇·(q**u**), `VCV` = −K ∂(qω)/∂p
- MERRA-2 physics tendencies: `MOKE`/`TOKE`/`GOKE` (moist, turbulence and
  gravity-wave drag on **u**), `POV`/`TOV`/`COV` (moist, turbulence and
  chemistry on q), `AOKEV` (analysis increment). `DOKE`/`DOV` use MERRA-2's own
  DYN tendency, which is a cross-check on HA+VA+PE.
- The `I` prefix marks the column integral. `SPTE` = (∂p_s/∂t)·KEV(p_bot)/g is
  the Leibniz boundary term.
- There are two tendencies. `IKEVTEND` differences the instantaneous IKEV over
  3 h. `I_KEVTEND` integrates K ∂q/∂t + q**u**·∂**u**/∂t.
- VTE budget, suffix `_v`: momentum terms weighted by q², vapor terms by 2qK.
  `HCV_v` = −K∇·(q²**u**) is the user's chosen form. The `_vO` pair is the
  NCL's commented-out −2qK∇·(q**u**) form.
- Grouping and residual: tend = hflux (HAKE+HCV) + vflux (VAKE+VCV) + pe + cond
  + fric + ana + spte + resid (`make_website_budget.py`). In MERRA-2 the
  residual is about 30% of the tendency; it is about 3% when MERRA-2's DYN
  tendencies replace the offline dynamics.

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
- Known small flaws, left as they are for now: metpy without a CRS
  double-counts the meridional ellipsoid factor (+0.2–0.7%); longitude
  derivatives are not cyclic at the ±180° seam. Fixing either is a deliberate
  change (see below).

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
- Separate the layers. Data-source adapters (names, units, level order,
  coordinates) produce a standard Dataset. Operators (integrate, grad, div,
  d/dp, interpolate) and budget-term functions are pure and source-agnostic.
- One function per physical term or small group of terms, named for the
  physics, with the equation and units in the docstring. Avoid the long
  monolithic `main()`.

## Numerical consistency (required)

- The package must reproduce the reference outputs to float32 rounding unless a
  change is deliberate. Report every deliberate change to the user with its
  size (relative RMS per term).
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
