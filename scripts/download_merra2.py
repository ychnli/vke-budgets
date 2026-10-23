"""
Download MERRA-2 for the VKE budgets: the five files per day that
vke_budgets.sources.merra2.open_day reads, cut down to the variables it
reads (merra2.VARIABLES), without the DYN tendencies unless --with-dyn.

    python scripts/download_merra2.py --start 2010-01-01 --end 2010-01-31 --root ROOT

fetches 2010-01-01 ... 2010-01-31, and the inst3_3d_asm_Np file of
2010-02-01, whose 00Z closes the last window of the 31st. The files go to
ROOT/YYYY/ under their original names, where merra2.find_files looks.

Each file is downloaded whole over HTTPS to a work directory (GES DISC's
cloud OPeNDAP cannot cut the tendency files before it times out), cut with
`nccopy -V`, which keeps the values, attributes, chunking and compression,
checked, and only then moved into ROOT. So a file in ROOT is complete, and
reruns skip it. Without DYN a day is 2.7 GB instead of 5.3 GB.

The file names come from NASA's CMR catalogue, which takes care of the
stream numbers (MERRA2_300 up to 2010, 400 after, 401 for reprocessed
months). Authentication: an Earthdata Login entry (machine
urs.earthdata.nasa.gov) in ~/.netrc. Needs curl and nccopy on the PATH.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import date, timedelta

import netCDF4

from vke_budgets.sources import merra2

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
VERSION = "5.12.4"
SHORT_NAMES = {"inst3_3d_asm_Np": "M2I3NPASM", "tavg3_3d_asm_Nv": "M2T3NVASM",
               "tavg3_3d_udt_Np": "M2T3NPUDT", "tavg3_3d_qdt_Np": "M2T3NPQDT"}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ATTEMPTS = 4


def variables(collection: str, with_dyn: bool) -> list[str]:
    """The variables kept from a collection, coordinates included."""
    names = [v for v in merra2.VARIABLES[collection]
             if with_dyn or not v.endswith(merra2.OPTIONAL_CODES)]
    return names + list(merra2.COORDINATES)


def granules(collection: str, first: str, last: str) -> dict[str, str]:
    """{YYYYMMDD: download URL} for one collection, days first..last
    (YYYYMMDD), from the CMR catalogue. Every day must have one file."""
    found = {}
    page = 1
    while True:
        query = urllib.parse.urlencode({
            "short_name": SHORT_NAMES[collection], "version": VERSION, "page_size": 2000,
            "page_num": page, "temporal": f"{first[:4]}-{first[4:6]}-{first[6:]}T00:00:00Z,"
                                          f"{last[:4]}-{last[4:6]}-{last[6:]}T23:59:59Z"})
        with urllib.request.urlopen(f"{CMR}?{query}", timeout=120) as response:
            entries = json.load(response)["feed"]["entry"]
        for entry in entries:
            for link in entry.get("links", []):
                url = link["href"]
                if url.startswith("https://") and url.endswith(".nc4") \
                        and link.get("rel", "").endswith("/data#"):
                    day = url.rsplit(".", 2)[-2]          # ...<collection>.YYYYMMDD.nc4
                    if first <= day <= last:
                        found.setdefault(day, set()).add(url)
        if len(entries) < 2000:
            break
        page += 1

    days = [d.strftime("%Y%m%d") for d in date_range(first, last)]
    missing = [d for d in days if d not in found]
    several = {d: sorted(u) for d, u in found.items() if len(u) > 1}
    if missing or several:
        raise RuntimeError(f"{collection}: no file for {missing}; several files for {several}")
    return {d: found[d].pop() for d in days}


def date_range(first: str, last: str) -> Iterator[date]:
    day, last = parse(first), parse(last)
    while day <= last:
        yield day
        day += timedelta(days=1)


def parse(text: str) -> date:
    text = text.replace("-", "")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def fetch(url: str, path: str, cookies: str) -> None:
    """Download url to path, following the Earthdata Login redirect."""
    subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--netrc",
                    "--cookie", cookies, "--cookie-jar", cookies, "--output", path, url],
                   check=True)
    with open(path, "rb") as f:
        if f.read(8) != HDF5_MAGIC:
            raise RuntimeError(f"{url} did not return a netCDF-4 file (a login or error page?)")


def cut(source: str, target: str, names: list[str]) -> None:
    """Copy only `names` from source to target, and check the result."""
    subprocess.run(["nccopy", "-V", ",".join(names), source, target], check=True)
    with netCDF4.Dataset(target) as f:
        absent = [v for v in names if v not in f.variables]
        if absent or f.dimensions["time"].size != 8:
            raise RuntimeError(f"{target}: missing {absent} or not 8 time steps")


def download(url: str, root: str, work: str, names: list[str], cookies: str) -> int:
    """One file into root/YYYY/, unless it is there already. Returns the
    size kept in bytes (0 if skipped)."""
    name = os.path.basename(url)
    final = os.path.join(root, name.rsplit(".", 2)[-2][:4], name)
    if os.path.exists(final):
        return 0
    os.makedirs(os.path.dirname(final), exist_ok=True)
    whole = os.path.join(work, f"{name}.{os.getpid()}")
    part = f"{final}.{os.getpid()}.part"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            start = time.time()
            fetch(url, whole, cookies)
            fetched = time.time()
            cut(whole, part, names)
            os.replace(part, final)
            print(f"{name}  {os.path.getsize(whole) / 1e9:.2f} -> "
                  f"{os.path.getsize(final) / 1e9:.2f} GB  fetch {fetched - start:.0f} s, "
                  f"cut {time.time() - fetched:.0f} s", flush=True)
            return os.path.getsize(final)
        except (subprocess.CalledProcessError, RuntimeError, OSError) as error:
            print(f"{name}: attempt {attempt} failed: {error}", file=sys.stderr, flush=True)
            if attempt < ATTEMPTS:
                time.sleep(60 * attempt)
        finally:
            for path in (whole, part):
                if os.path.exists(path):
                    os.remove(path)
    raise RuntimeError(f"{name}: gave up after {ATTEMPTS} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last day, YYYY-MM-DD")
    parser.add_argument("--root", required=True, help="output root, one directory per year")
    parser.add_argument("--work", help="directory for the whole files while they are cut "
                                       "(one at a time, up to 2.3 GB); default ROOT/.work")
    parser.add_argument("--with-dyn", action="store_true",
                        help="keep the DYN tendencies (for BudgetOptions.dyn_crosscheck)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the files and their streams, download nothing")
    args = parser.parse_args()

    first, last = parse(args.start), parse(args.end)
    after = (last + timedelta(days=1)).strftime("%Y%m%d")
    first, last = first.strftime("%Y%m%d"), last.strftime("%Y%m%d")
    urls = {collection: granules(collection, first,
                                 after if collection == "inst3_3d_asm_Np" else last)
            for collection in SHORT_NAMES}
    if args.dry_run:
        for collection, by_day in urls.items():
            streams = sorted({os.path.basename(u).split(".")[0] for u in by_day.values()})
            print(f"{collection}: {len(by_day)} files, {min(by_day)}..{max(by_day)}, {streams}")
        return

    work = args.work or os.path.join(args.root, ".work")
    os.makedirs(work, exist_ok=True)
    cookies = os.path.join(work, f"cookies.{os.getpid()}")
    failed, kept = [], 0
    try:
        for day in sorted({d for by_day in urls.values() for d in by_day}):
            for collection, by_day in urls.items():
                if day in by_day:
                    try:
                        kept += download(by_day[day], args.root, work,
                                         variables(collection, args.with_dyn), cookies)
                    except RuntimeError as error:
                        failed.append(str(error))
    finally:
        if os.path.exists(cookies):
            os.remove(cookies)
    print(f"{args.start}..{args.end}: {kept / 1e9:.1f} GB written, {len(failed)} failed")
    if failed:
        sys.exit("\n".join(failed))


if __name__ == "__main__":
    main()
