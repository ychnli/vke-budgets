"""
Download a monthly MERRA-2 collection, cut to chosen variables: by default
the monthly mean diurnal cycle of humidity and wind on pressure levels.

    python scripts/download_merra2_monthly.py --start 1980-01 --end 2026-09 --root ROOT

fetches MERRA2_*.instU_3d_asm_Np.YYYYMM.nc4 for 1980-01 ... 2026-09 into
ROOT, keeping QV, U and V. Each instU file holds one month: the mean of the
instantaneous state at 00, 03, ..., 21Z over the days of that month (8 time
steps), on the 42 pressure levels. Averaging these files over years gives a
climatology of the diurnal cycle per calendar month.

As in download_merra2.py (whose fetch and checks this reuses), each file is
downloaded whole to a work directory, cut with `nccopy -V`, checked, and
only then moved into ROOT, so reruns skip the files already there. Several
files are fetched at once (--workers). Authentication: an Earthdata Login
entry in ~/.netrc.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import netCDF4

import download_merra2 as daily

SHORT_NAMES = {"instU_3d_asm_Np": "M2IUNPASM", "instM_3d_asm_Np": "M2IMNPASM",
               "tavgU_3d_udt_Np": "M2TUNPUDT", "tavgM_3d_udt_Np": "M2TMNPUDT",
               "tavgU_3d_qdt_Np": "M2TUNPQDT", "tavgM_3d_qdt_Np": "M2TMNPQDT"}
COORDINATES = ("time", "lev", "lat", "lon")


def months(first: str, last: str) -> list[str]:
    """YYYYMM from first to last (YYYY-MM), inclusive."""
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y:04d}{m:02d}" <= last.replace("-", ""):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def granules(collection: str, wanted: list[str]) -> dict[str, str]:
    """{YYYYMM: download URL} from the CMR catalogue. Every month must have
    exactly one file."""
    found = {}
    page = 1
    while True:
        query = urllib.parse.urlencode({
            "short_name": SHORT_NAMES[collection], "version": daily.VERSION,
            "page_size": 2000, "page_num": page,
            "temporal": f"{wanted[0][:4]}-{wanted[0][4:]}-01T00:00:00Z,"
                        f"{wanted[-1][:4]}-{wanted[-1][4:]}-28T23:59:59Z"})
        with urllib.request.urlopen(f"{daily.CMR}?{query}", timeout=120) as response:
            entries = json.load(response)["feed"]["entry"]
        for entry in entries:
            for link in entry.get("links", []):
                url = link["href"]
                if url.startswith("https://") and url.endswith(".nc4") \
                        and link.get("rel", "").endswith("/data#"):
                    month = url.rsplit(".", 2)[-2]          # ...<collection>.YYYYMM.nc4
                    if month in wanted:
                        found.setdefault(month, set()).add(url)
        if len(entries) < 2000:
            break
        page += 1
    missing = [m for m in wanted if m not in found]
    several = {m: sorted(u) for m, u in found.items() if len(u) > 1}
    if missing or several:
        raise RuntimeError(f"{collection}: no file for {missing}; several files for {several}")
    return {m: found[m].pop() for m in wanted}


def cut(source: str, target: str, names: list[str]) -> None:
    """Copy only `names` from source to target, and check the result."""
    subprocess.run(["nccopy", "-V", ",".join(names), source, target], check=True)
    with netCDF4.Dataset(target) as f:
        absent = [v for v in names if v not in f.variables]
        if absent:
            raise RuntimeError(f"{target}: missing {absent}")


def download(url: str, root: str, work: str, names: list[str]) -> int:
    """One file into root, unless it is there already. Returns the size kept
    in bytes (0 if skipped)."""
    name = os.path.basename(url)
    final = os.path.join(root, name)
    if os.path.exists(final):
        return 0
    tag = f"{os.getpid()}.{threading.get_ident()}"
    whole = os.path.join(work, f"{name}.{tag}")
    part = f"{final}.{tag}.part"
    cookies = os.path.join(work, f"cookies.{tag}")
    for attempt in range(1, daily.ATTEMPTS + 1):
        try:
            start = time.time()
            daily.fetch(url, whole, cookies)
            fetched = time.time()
            cut(whole, part, names)
            os.replace(part, final)
            print(f"{name}  {os.path.getsize(whole) / 1e9:.2f} -> "
                  f"{os.path.getsize(final) / 1e9:.2f} GB  fetch {fetched - start:.0f} s, "
                  f"cut {time.time() - fetched:.0f} s", flush=True)
            return os.path.getsize(final)
        except (subprocess.CalledProcessError, RuntimeError, OSError) as error:
            print(f"{name}: attempt {attempt} failed: {error}", file=sys.stderr, flush=True)
            if attempt < daily.ATTEMPTS:
                time.sleep(60 * attempt)
        finally:
            for path in (whole, part, cookies):
                if os.path.exists(path):
                    os.remove(path)
    raise RuntimeError(f"{name}: gave up after {daily.ATTEMPTS} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM")
    parser.add_argument("--root", required=True, help="output directory")
    parser.add_argument("--collection", default="instU_3d_asm_Np", choices=sorted(SHORT_NAMES))
    parser.add_argument("--variables", default="QV,U,V",
                        help="comma-separated variables to keep (coordinates are kept)")
    parser.add_argument("--work", help="directory for the whole files while they are cut; "
                                       "default ROOT/.work")
    parser.add_argument("--workers", type=int, default=4, help="files fetched at once")
    args = parser.parse_args()

    names = args.variables.split(",") + list(COORDINATES)
    urls = granules(args.collection, months(args.start, args.end))
    os.makedirs(args.root, exist_ok=True)
    work = args.work or os.path.join(args.root, ".work")
    os.makedirs(work, exist_ok=True)

    failed, kept = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(download, url, args.root, work, names): m for m, url in urls.items()}
        for job, month in jobs.items():
            try:
                kept += job.result()
            except RuntimeError as error:
                failed.append(str(error))
    print(f"{args.collection} {args.start}..{args.end}: {kept / 1e9:.1f} GB written, "
          f"{len(failed)} failed")
    if failed:
        sys.exit("\n".join(failed))


if __name__ == "__main__":
    main()
