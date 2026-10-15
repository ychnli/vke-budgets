"""
Run the budgets over a range of days and write them to disk, resumably.

One call (or one Slurm job) handles a contiguous range of days, normally a
month or more; a climatology is a handful of such jobs, not one job per day.

Clean layout: the output is one file per calendar month (or the part of
one that the range covers), named <prefix>_<first day>-<last day>.nc. Each
finished day is first written to <out_dir>/parts/<day>.nc, so a job that is
killed resumes at the day it was on. When a month is complete its parts are
merged into the month file, rounded to `significant_bits` if given, and
deleted. Months whose file exists are skipped.

Legacy layout: the reference's two files per day; days whose files exist
are skipped.
"""
import glob
import os
import time

import pandas as pd

from . import io
from .budget import BudgetOptions, compute
from .sources import merra2


def _month_ranges(start, end):
    """[(first, last), ...] for each calendar month that [start, end] touches."""
    days = pd.date_range(start, end, freq="D")
    months = days.to_period("M")
    return [(days[months == m][0], days[months == m][-1]) for m in months.unique()]


def period_path(out_dir, first, last, prefix="budget"):
    return os.path.join(out_dir, f"{prefix}_{first:%Y%m%d}-{last:%Y%m%d}.nc")


def run(start, end, root, out_dir, options=BudgetOptions(), layout="clean",
        significant_bits=None, n_workers=1, prefix="budget", open_day=merra2.open_day,
        log=None):
    """Compute and write the budgets for every day from `start` to `end`
    (inclusive, anything pandas reads as a date).

    Args:
        root: input root, passed to open_day(date, root).
        out_dir: output directory, created if needed.
        options: BudgetOptions.
        layout: "clean" or "legacy" (see the module docstring).
        significant_bits: BitRound precision of the month files (None keeps
            float32 exactly). Parts are always written exactly.
        n_workers: windows computed at once within a day.
        open_day: the source adapter, a function (YYYYMMDD, root) -> standard
            input Dataset.
        log: function taking a progress message; by default, print.
    """
    log = log or (lambda message: print(message, flush=True))
    os.makedirs(out_dir, exist_ok=True)

    def one_day(day):
        t0 = time.time()
        result = compute(open_day(f"{day:%Y%m%d}", root), options, n_workers=n_workers)
        log(f"{day:%Y-%m-%d} computed in {time.time() - t0:.0f} s")
        return result

    if layout == "legacy":
        for day in pd.date_range(start, end, freq="D"):
            if all(os.path.exists(p) for p in io.legacy_paths(out_dir, f"{day:%Y%m%d}")):
                continue
            io.write_legacy(one_day(day), out_dir, day)
        return
    if layout != "clean":
        raise ValueError(f"unknown layout {layout!r}")

    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    for first, last in _month_ranges(start, end):
        path = period_path(out_dir, first, last, prefix)
        if os.path.exists(path):
            log(f"{os.path.basename(path)} exists, skipped")
            continue
        parts = []
        for day in pd.date_range(first, last, freq="D"):
            part = os.path.join(parts_dir, f"{day:%Y%m%d}.nc")
            if not os.path.exists(part):
                io.write_period(one_day(day), part)
            parts.append(part)
        with io.open_budget(parts) as month:
            month.attrs = {**month.attrs, "first_day": f"{first:%Y-%m-%d}",
                           "last_day": f"{last:%Y-%m-%d}"}
            io.write_period(month, path, significant_bits=significant_bits)
        for part in parts:
            os.remove(part)
        log(f"wrote {path}")
    if not glob.glob(os.path.join(parts_dir, "*")):
        os.rmdir(parts_dir)
