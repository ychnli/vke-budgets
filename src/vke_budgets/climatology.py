"""
AR climatology: how often each grid point lies inside an atmospheric river.

For every instantaneous time step, the column fields ivt, kev and vte
(budget.instantaneous_columns) are computed and ARs are detected in each
(detection.detect). The masks then reduce to statistics per grid point:

    ar_count_<field>      number of time steps with an AR at the point
    ar_frequency_<field>  ar_count_<field> / n_steps
    mean_<field>          mean of the column field
    n_steps               number of time steps

The AR frequency is the fraction of time steps with an AR at a point (Ong &
Yang 2024, Methods); times 365.25 it is in days per year, the units of the
frequency maps of Zhang et al. (2026, Figure 1). Counts and n_steps add up
over periods, so the frequency of many months is

    sum over months of ar_count / sum over months of n_steps.

`run` works one calendar month at a time and writes one file per month,
<out_dir>/ar_stats_YYYYMM.nc, and with save_steps also the time steps
themselves (the columns and the masks), <out_dir>/ar_steps_YYYYMM.nc.

Each day contributes the instants of its own standard input Dataset except
the last, which closes the day's last window and is the first instant of the
next day (00Z for MERRA-2). So no instant is counted twice, and a MERRA-2
month has 8 steps per day.

Detection runs once per month and field, not once per day: the masks are the
same, because DetectBlobs treats every time step on its own, but each
DetectBlobs run costs about 13 s to start, whatever the number of steps.
"""
import calendar
import os
import time
from collections.abc import Callable, Iterable
from datetime import date

import numpy as np
import xarray as xr

from . import detection, io
from .budget import BudgetOptions, instantaneous_columns
from .sources import merra2

FIELDS = ("ivt", "kev", "vte")
# The part of the instantaneous state that the columns need.
INST_INPUTS = ["q_inst", "u_inst", "v_inst"]


def own_columns(ds: xr.Dataset, options: BudgetOptions = BudgetOptions()) -> xr.Dataset:
    """ivt, kev and vte of a standard input Dataset at all its instants but
    the last (see the module docstring), on (time_inst, lat, lon)."""
    # Load before integrating: integrating the lazy fields is twice as slow.
    inst = ds[INST_INPUTS].isel(time_inst=slice(0, -1)).load()
    return instantaneous_columns(inst, options).assign_attrs(source=ds.attrs.get("source", ""))


def period_columns(days: Iterable[str], root: str, options: BudgetOptions = BudgetOptions(),
                   open_day: Callable[[str, str], xr.Dataset] = merra2.open_day,
                   log: Callable[[str], None] | None = None) -> xr.Dataset:
    """own_columns of each day (YYYYMMDD), joined along time_inst. open_day
    is the source adapter, (YYYYMMDD, root) -> standard input Dataset; it
    should fail if a day's files are missing."""
    log = log or _print
    parts = []
    for day in days:
        start = time.time()
        parts.append(own_columns(open_day(day, root), options))
        log(f"{day} columns in {time.time() - start:.0f} s")
    return xr.concat(parts, "time_inst")


def ar_statistics(columns: xr.Dataset, masks: xr.Dataset) -> xr.Dataset:
    """AR counts, AR frequencies and mean columns per grid point, for every
    field that has a mask (see the module docstring)."""
    n_steps = columns.sizes["time_inst"]
    stats = {"n_steps": xr.DataArray(np.int32(n_steps), attrs={
        "long_name": "number of time steps", "units": "1"})}
    for field in FIELDS:
        if f"mask_{field}" not in masks:
            continue
        mask = masks[f"mask_{field}"]
        count = mask.sum("time_inst").astype(np.int32)
        stats[f"ar_count_{field}"] = count.assign_attrs(
            long_name=f"number of time steps with an AR detected from {field}", units="1")
        stats[f"ar_frequency_{field}"] = (count / n_steps).assign_attrs(
            long_name=f"fraction of time steps with an AR detected from {field}", units="1",
            detection=mask.attrs.get("long_name", ""))
        stats[f"mean_{field}"] = columns[field].mean("time_inst").assign_attrs(
            {**columns[field].attrs,
             "long_name": "mean of the " + columns[field].attrs.get("long_name", field)})
    return xr.Dataset(stats)


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


def task_months(months: list[date], task: int, months_per_task: int = 1) -> list[date]:
    """The task-th group of months_per_task consecutive months (task 0 is the
    first), for array jobs. The last group may be shorter."""
    n_tasks = -(-len(months) // months_per_task)          # ceil(months / per task)
    if not 0 <= task < n_tasks:
        raise ValueError(f"task {task}: {len(months)} months in groups of {months_per_task} "
                         f"make tasks 0 to {n_tasks - 1}")
    return months[task * months_per_task:(task + 1) * months_per_task]


def month_paths(out_dir: str, month: date) -> tuple[str, str]:
    """(statistics file, time-step file) of a month."""
    return (os.path.join(out_dir, f"ar_stats_{month:%Y%m}.nc"),
            os.path.join(out_dir, f"ar_steps_{month:%Y%m}.nc"))


def run(start: str, end: str, root: str, out_dir: str, task: int | None = None,
        months_per_task: int = 1, save_steps: bool = False,
        options: BudgetOptions = BudgetOptions(),
        thresholds: dict[str, float] = detection.REFERENCE_THRESHOLDS,
        open_day: Callable[[str, str], xr.Dataset] = merra2.open_day,
        log: Callable[[str], None] | None = None) -> None:
    """Compute and write the AR statistics of every month from start to end
    (YYYY-MM, inclusive), or of the task-th group of months_per_task months.

    Months whose files exist are skipped, so reruns resume.

    Args:
        root: input root, passed to open_day(YYYYMMDD, root).
        out_dir: output directory, created if needed.
        save_steps: also write the columns and masks of every time step.
        options: for the column integrals (g, p_bottom, p_top, quantities).
        thresholds: Laplacian threshold of each field (detection.detect).
        open_day: the source adapter.
        log: function taking a progress message; by default, print.
    """
    log = log or _print
    months = months_between(start, end)
    if task is not None:
        months = task_months(months, task, months_per_task)
    os.makedirs(out_dir, exist_ok=True)

    for month in months:
        stats_path, steps_path = month_paths(out_dir, month)
        if os.path.exists(stats_path) and (not save_steps or os.path.exists(steps_path)):
            log(f"{month:%Y-%m} exists, skipped")
            continue
        columns = period_columns(days_in_month(month), root, options, open_day, log)
        t0 = time.time()
        masks = detection.detect(columns, thresholds)
        log(f"{month:%Y-%m} detection in {time.time() - t0:.0f} s")
        attrs = {"source": columns.attrs.get("source", ""), "month": f"{month:%Y-%m}",
                 "detection": "TempestExtremes DetectBlobs (vke_budgets.detection.detect)",
                 "created_by": "vke_budgets.climatology.run"}
        if save_steps:
            _write(xr.merge([columns, masks]).assign_attrs(attrs), steps_path)
        _write(ar_statistics(columns, masks).assign_attrs(attrs), stats_path)
        log(f"wrote {stats_path}")


def _write(ds: xr.Dataset, path: str) -> None:
    """Compressed, with floats as float32 and masks as bytes (0 or 1)."""
    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dtype == bool:
            dtype = "int8"
        elif np.issubdtype(da.dtype, np.floating):
            dtype = "float32"
        else:
            dtype = da.dtype
        encoding[name] = {"dtype": dtype, "zlib": True, "complevel": 1}
    io._write_atomic(ds, path, encoding)


def _print(message: str) -> None:
    print(message, flush=True)
