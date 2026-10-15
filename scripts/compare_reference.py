#!/usr/bin/env python
"""
Compute one day of the budgets and compare every variable with the
reference outputs (analysis_AR_IVKE1_{tavg,inst}<date>.nc).

    python scripts/compare_reference.py --date 20170712 \\
        --root $GROUP_SCRATCH/Aidi/VKE_budget/data \\
        --ref-dir $GROUP_SCRATCH/Aidi/VKE_budget/case2023/verify/port_20170712 \\
        --config reference

Statistics are over |lat| < 87 and all time steps: relative RMS
RMS(ours - ref) / RMS(ref), max |ours - ref|, and the number of points that
are missing in one and not the other.

Configurations (--config, repeatable): `reference` (geometry.REFERENCE,
which should match to float32 rounding), `default` (the package defaults),
and one fix at a time on top of REFERENCE, to size each deliberate change:
`fix-dy` (no extra a/M), `fix-metric` (metric term with N), `fix-seam`
(periodic longitude), `fix-all` (all three, still WGS84), and `sphere`
(= default).
"""
import argparse
import os
import resource
import time
from dataclasses import replace

import numpy as np
import xarray as xr

from vke_budgets import budget
from vke_budgets.geometry import DEFAULT, REFERENCE
from vke_budgets.sources import merra2

CONFIGS = {
    "reference": REFERENCE,
    "fix-dy": replace(REFERENCE, metpy_dy_factor=False),
    "fix-metric": replace(REFERENCE, metric_radius=None),
    "fix-seam": replace(REFERENCE, periodic_lon=None),
    "fix-all": replace(REFERENCE, metpy_dy_factor=False, metric_radius=None, periodic_lon=None),
    "sphere": DEFAULT,
    "default": DEFAULT,
}


def stats(ours, ref, inner):
    a = np.asarray(ours.where(inner), dtype=np.float64)
    b = np.asarray(ref.where(inner), dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    diff = a[ok] - b[ok]
    rms_b = np.sqrt(np.mean(b[ok] ** 2))
    return (np.sqrt(np.mean(diff**2)) / rms_b if rms_b > 0 else np.nan,
            np.abs(diff).max(), int((np.isfinite(a) != np.isfinite(b)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--root", required=True, help="MERRA-2 root, one directory per year")
    ap.add_argument("--ref-dir", required=True, help="directory of the reference outputs")
    ap.add_argument("--config", action="append", choices=CONFIGS, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save", help="write our output for each config to SAVE_<config>.nc")
    args = ap.parse_args()

    ref = xr.open_dataset(os.path.join(args.ref_dir, f"analysis_AR_IVKE1_tavg{args.date}.nc"))
    ref_inst = xr.open_dataset(os.path.join(args.ref_dir, f"analysis_AR_IVKE1_inst{args.date}.nc"))
    ds = merra2.open_day(args.date, args.root)

    for config in args.config or ["reference"]:
        options = budget.BudgetOptions(geometry=CONFIGS[config], dyn_crosscheck=True,
                                       ncl_vte_form=True, flux_form=True)
        t0 = time.time()
        out = budget.compute(ds, options, n_workers=args.workers)
        elapsed = time.time() - t0
        if args.save:
            out.to_netcdf(f"{args.save}_{config}.nc")

        inner = np.abs(out["lat"]) < 87
        print(f"\n== {args.date}, config {config}: {elapsed:.0f} s, "
              f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
        print(f"{'variable':12s} {'rel RMS':>9s} {'max |diff|':>11s} {'NaN mismatch':>13s}")
        common = sorted(set(out.data_vars) & set(ref.data_vars))
        for name in common:
            rel, mx, nan = stats(out[name].transpose(*ref[name].dims), ref[name], inner)
            print(f"{name:12s} {rel:9.2e} {mx:11.3e} {nan:13d}")
        n_inst = ref_inst.sizes["time"]
        for name in ("IKEV", "IVTE", "IVT"):
            ours = out[name].isel(time_inst=slice(0, n_inst)).rename(time_inst="time")
            ours = ours.assign_coords(time=ref_inst["time"].values)
            rel, mx, nan = stats(ours, ref_inst[name], np.abs(ref_inst["lat"]) < 90)
            print(f"{name + ' (inst)':12s} {rel:9.2e} {mx:11.3e} {nan:13d}")
        print("not in the reference:", sorted(set(out.data_vars) - set(ref.data_vars) - {"IKEV", "IVTE", "IVT"}))
        print("not computed here:", sorted(set(ref.data_vars) - set(out.data_vars)))


if __name__ == "__main__":
    main()
