"""
Check which days of MERRA-2 under a root directory are complete for the
budget, and summarize them by year and month.

    python scripts/check_merra2.py --root ROOT --start 1980-01-01 --end 2026-09-30
    python scripts/check_merra2.py --root ROOT --start 1980-01-01 --end 2026-09-30 --with-dyn

A budget day D needs the five files that merra2.find_files looks up:
inst3_3d_asm_Np of D and of D+1 (its 00Z closes the last window of D), and
tavg3_3d_asm_Nv, tavg3_3d_udt_Np and tavg3_3d_qdt_Np of D, each as
ROOT/YYYY/MERRA2_<stream>.<collection>.<YYYYMMDD>.nc4. A day is complete
when each of them is there exactly once (two streams of one file make
find_files fail).

The check goes by file name, listing each year directory once. Files still
being downloaded (*.part) don't count. With --with-dyn, it also opens the
udt and qdt files and checks that they hold the DYN tendencies, which
BudgetOptions.dyn_crosscheck needs: files cut without them have the same
names. That opens two files per day, so it takes a few minutes for decades.

Prints one line per year (complete days, complete months, and the files
missing per collection); --missing also lists the incomplete days. Exits
with status 1 if any day is incomplete.
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

import netCDF4

from vke_budgets.sources import merra2

NAME = re.compile(r"^MERRA2_\d{3}\.(?P<collection>\w+)\.(?P<day>\d{8})\.nc4$")
TENDENCIES = ("tavg3_3d_udt_Np", "tavg3_3d_qdt_Np")
LABELS = {"inst3_3d_asm_Np": "inst", "tavg3_3d_asm_Nv": "Nv",
          "tavg3_3d_udt_Np": "udt", "tavg3_3d_qdt_Np": "qdt"}


def listing(root: str, years: range) -> dict[tuple[str, str], list[str]]:
    """{(collection, YYYYMMDD): [paths]} for every MERRA-2 file in the
    year directories of root."""
    found = defaultdict(list)
    for year in years:
        directory = os.path.join(root, str(year))
        if not os.path.isdir(directory):
            continue
        for entry in os.scandir(directory):
            match = NAME.match(entry.name)
            if match:
                found[match["collection"], match["day"]].append(entry.path)
    return found


def needed(day: date) -> list[tuple[str, str]]:
    """The (collection, YYYYMMDD) of the five files of a budget day."""
    after = day + timedelta(days=1)
    return [(collection, (after if key == "inst_next" else day).strftime("%Y%m%d"))
            for key, collection in merra2.COLLECTIONS.items()]


def has_dyn(path: str, collection: str) -> bool:
    """Whether a udt or qdt file holds its DYN tendencies."""
    names = [v for v in merra2.VARIABLES[collection] if v.endswith(merra2.OPTIONAL_CODES)]
    with netCDF4.Dataset(path) as f:
        return all(v in f.variables for v in names)


def check(root: str, first: date, last: date, with_dyn: bool = False) -> dict[date, list[str]]:
    """{day: problems} for every day first..last; an empty list means the
    day is complete. A problem reads e.g. 'qdt 19900101 missing'."""
    found = listing(root, range(first.year, last.year + 2))
    dyn_ok = {}
    problems = {}
    day = first
    while day <= last:
        problems[day] = []
        for collection, stamp in needed(day):
            paths = found.get((collection, stamp), [])
            label = f"{LABELS[collection]} {stamp}"
            if not paths:
                problems[day].append(f"{label} missing")
            elif len(paths) > 1:
                problems[day].append(f"{label} in {len(paths)} streams")
            elif with_dyn and collection in TENDENCIES:
                if paths[0] not in dyn_ok:
                    dyn_ok[paths[0]] = has_dyn(paths[0], collection)
                if not dyn_ok[paths[0]]:
                    problems[day].append(f"{label} without DYN")
        day += timedelta(days=1)
    return problems


def summary(problems: dict[date, list[str]]) -> list[str]:
    """One line per year: complete days and months, and the problems per
    collection."""
    lines = [f"{'year':>4} {'days':>5} {'complete':>9} {'months':>7}   "
             + " ".join(f"{label:>5}" for label in LABELS.values()) + "   (files with problems)"]
    for year in sorted({d.year for d in problems}):
        days = [d for d in problems if d.year == year]
        complete = [d for d in days if not problems[d]]
        months = sorted({d.month for d in days})
        full = [m for m in months if all(not problems[d] for d in days if d.month == m)]
        # Count each file once (inst of the 1st is needed by two days).
        bad = {p for d in days for p in problems[d]}
        counts = [sum(p.startswith(label + " ") for p in bad) for label in LABELS.values()]
        lines.append(f"{year:>4} {len(days):>5} {len(complete):>9} {len(full):>3}/{len(months):<3}   "
                     + " ".join(f"{c:>5}" for c in counts))
    return lines


def ranges(days: list[date]) -> list[tuple[date, date]]:
    """Consecutive days grouped as (first, last)."""
    out = []
    for day in sorted(days):
        if out and day - out[-1][1] == timedelta(days=1):
            out[-1] = (out[-1][0], day)
        else:
            out.append((day, day))
    return out


def parse(text: str) -> date:
    text = text.replace("-", "")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="MERRA-2 root, one directory per year")
    parser.add_argument("--start", required=True, help="first budget day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last budget day, YYYY-MM-DD")
    parser.add_argument("--with-dyn", action="store_true",
                        help="also require the DYN tendencies in the udt and qdt files")
    parser.add_argument("--missing", action="store_true",
                        help="also list the incomplete days and what they lack")
    args = parser.parse_args()

    problems = check(args.root, parse(args.start), parse(args.end), args.with_dyn)
    print("\n".join(summary(problems)))
    incomplete = [d for d, p in problems.items() if p]
    years = sorted({d.year for d in problems})
    full_years = [y for y in years if not any(problems[d] for d in problems if d.year == y)]
    print(f"\n{len(problems) - len(incomplete)} of {len(problems)} days complete"
          f"{' (with DYN)' if args.with_dyn else ''}; "
          f"{len(full_years)} complete years: {', '.join(map(str, full_years)) or 'none'}")
    if args.missing and incomplete:
        print("\nincomplete days:")
        for a, b in ranges(incomplete):
            lacking = sorted({p.split()[0] + " " + " ".join(p.split()[2:])
                              for d in problems if a <= d <= b for p in problems[d]})
            print(f"  {a}..{b} ({(b - a).days + 1} d): {', '.join(lacking)}")
    sys.exit(1 if incomplete else 0)


if __name__ == "__main__":
    main()
