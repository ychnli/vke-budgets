"""
AR climatology from MERRA-2, one calendar month at a time: the column fields
IVT, KEV and VTE at every 3-hourly instant, the ARs detected from each
(detection.detect), and their monthly statistics. This is the input to the
AR frequency maps of Zhang et al. (2026), Figure 1.

    python scripts/compute_climatology.py --start 2010-01 --end 2019-12 \
        --root ROOT --out OUT [--task I [--months-per-task N]] [--save-steps]

For each month of the range it writes OUT/ar_stats_YYYYMM.nc. With --task,
only the I-th group of N consecutive months (counting from 0; N = 1 by
default) is done, for array jobs. The file holds

    ar_count_<field>      number of steps with an AR at the point
    ar_frequency_<field>  ar_count_<field> / n_steps
    mean_<field>          monthly mean of the column field
    n_steps               number of 3-hourly steps in the month

for field = ivt, kev, vte. With --save-steps it also writes
OUT/ar_steps_YYYYMM.nc, the 3-hourly columns and masks themselves (about
0.5 GB per month). Months whose files exist are skipped, so reruns resume.

The AR frequency is the fraction of time steps with an AR at a point (Ong &
Yang 2024, Methods). Over many months it is sum(ar_count) / sum(n_steps);
times 365.25 it is in days per year, the units of Figure 1.

Each day contributes its 8 instants 00, 03, ..., 21Z. The 00Z of the next
day, which open_day also returns, is the first instant of that day, so no
instant is counted twice and a month has 8 steps per day.
"""
import argparse
import calendar
import os
import time
from datetime import date

import numpy as np
import xarray as xr

from vke_budgets import budget, detection
from vke_budgets.sources import merra2

FIELDS = ("ivt", "kev", "vte")
# The instantaneous state that the columns need.
INST_INPUTS = ["q_inst", "u_inst", "v_inst"]


def months_between(start: str, end: str) -> list[date]:
    """The first day of every month from start to end (YYYY-MM), inclusive."""
    first, last = (date(int(s[:4]), int(s[5:7]), 1) for s in (start, end))
    months = []
    while first <= last:
        months.append(first)
        first = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return months


def days_in_month(month: date) -> list[str]:
    """Every day of the month, as YYYYMMDD."""
    n_days = calendar.monthrange(month.year, month.month)[1]
    return [f"{month:%Y%m}{day:02d}" for day in range(1, n_days + 1)]


def compute_month(month: date, root: str) -> tuple[xr.Dataset, xr.Dataset]:
    """The instantaneous columns and AR masks of every 3-hourly step of a
    month.

    For each day: open the five MERRA-2 files (open_day fails if one is
    missing), read the day's own 8 instants of the instantaneous state (the
    other collections are never read), and integrate ivt, kev and vte over
    the column (budget.instantaneous_columns). Then detect ARs in each field
    for the whole month at once (detection.detect).

    Detecting once per month gives the same masks as day by day, because
    DetectBlobs treats every time step on its own, but is much faster: each
    DetectBlobs call costs about 13 s to start, whatever the number of steps.

    Returns:
        (columns, masks): ivt, kev, vte and mask_ivt, mask_kev, mask_vte,
        each on (time_inst, lat, lon).
    """
    month_cols = []
    for day_str in days_in_month(month):
        start = time.time()
        day_data = merra2.open_day(day_str, root)
        # Load before integrating: integrating the lazy fields is twice as slow.
        inst = day_data[INST_INPUTS].isel(time_inst=slice(0, 8)).load()
        month_cols.append(budget.instantaneous_columns(inst))
        print(f"{day_str} columns in {time.time() - start:.0f} s", flush=True)
    columns = xr.concat(month_cols, "time_inst")
    start = time.time()
    masks = detection.detect(columns)
    print(f"{month:%Y-%m} detection in {time.time() - start:.0f} s", flush=True)
    return columns, masks


def monthly_statistics(columns: xr.Dataset, masks: xr.Dataset) -> xr.Dataset:
    """AR counts, AR frequencies and mean columns of one month, per grid point
    (see the module docstring)."""
    n_steps = columns.sizes["time_inst"]
    stats = {"n_steps": xr.DataArray(np.int32(n_steps), attrs={
        "long_name": "number of 3-hourly time steps", "units": "1"})}
    for field in FIELDS:
        mask = masks[f"mask_{field}"]
        count = mask.sum("time_inst").astype(np.int32)
        stats[f"ar_count_{field}"] = count.assign_attrs(
            long_name=f"number of time steps with an AR detected from {field}", units="1")
        stats[f"ar_frequency_{field}"] = (count / n_steps).assign_attrs(
            long_name=f"fraction of time steps with an AR detected from {field}", units="1",
            detection=mask.attrs["long_name"])
        stats[f"mean_{field}"] = columns[field].mean("time_inst").assign_attrs(
            {**columns[field].attrs, "long_name": "monthly mean of the "
             + columns[field].attrs["long_name"]})
    return xr.Dataset(stats)


def write(ds: xr.Dataset, path: str) -> None:
    """Write compressed, float as float32 and masks as bytes, to a temporary
    name first, so that an interrupted job never leaves a file that looks
    complete."""
    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dtype == bool:
            dtype = "int8"
        elif np.issubdtype(da.dtype, np.floating):
            dtype = "float32"
        else:
            dtype = da.dtype
        encoding[name] = {"dtype": dtype, "zlib": True, "complevel": 1}
    tmp = f"{path}.tmp"
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    parser.add_argument("--root", required=True, help="MERRA-2 root, one directory per year")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--task", type=int, default=None,
                        help="process only the task-th group of months in the range "
                             "(0 = the first; e.g. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--months-per-task", type=int, default=1,
                        help="months in each group of --task (default 1)")
    parser.add_argument("--save-steps", action="store_true",
                        help="also write the 3-hourly columns and masks")
    args = parser.parse_args()

    months = months_between(args.start, args.end)
    if args.task is not None:
        per_task = args.months_per_task
        n_tasks = -(-len(months) // per_task)           # ceil(months / per_task)
        if not 0 <= args.task < n_tasks:
            parser.error(f"--task {args.task}: {len(months)} months in groups of "
                         f"{per_task} make tasks 0 to {n_tasks - 1}")
        months = months[args.task * per_task:(args.task + 1) * per_task]
    os.makedirs(args.out, exist_ok=True)

    for month in months:
        stats_path = os.path.join(args.out, f"ar_stats_{month:%Y%m}.nc")
        steps_path = os.path.join(args.out, f"ar_steps_{month:%Y%m}.nc")
        if os.path.exists(stats_path) and (not args.save_steps or os.path.exists(steps_path)):
            print(f"{month:%Y-%m} exists, skipped", flush=True)
            continue
        columns, masks = compute_month(month, args.root)
        attrs = {"source": "MERRA-2", "month": f"{month:%Y-%m}",
                 "detection": "TempestExtremes DetectBlobs via vke_budgets.detection.detect",
                 "created_by": "scripts/compute_climatology.py"}
        if args.save_steps:
            write(xr.merge([columns, masks]).assign_attrs(attrs), steps_path)
        write(monthly_statistics(columns, masks).assign_attrs(attrs), stats_path)
        print(f"wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
