"""
Tropical cyclone tracks from IBTrACS v04r01 (International Best Track Archive
for Climate Stewardship, NOAA NCEI), to exclude TC cores from AR masks.

A track point counts as a tropical cyclone when IBTrACS gives its nature as
"TS" (tropical system), whatever its intensity: tropical depressions count,
the extratropical ("ET"), subtropical ("SS"), disturbance ("DS"), not
reported ("NR") and mixed ("MX", the agencies disagree) stages do not. Only
main tracks are kept: a spur track repeats points of a main track where two
storms merge or split.
"""
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

VERSION = "v04r01"
FILENAME = f"IBTrACS.since1980.{VERSION}.nc"
URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-"
       f"ibtracs/{VERSION}/access/netcdf/{FILENAME}")
TROPICAL = b"TS"


def default_directory() -> Path:
    """Where the tracks file is kept: $VKE_IBTRACS_DIR if set, else data/ibtracs
    in the repository (gitignored) for an editable install, else
    ~/.cache/vke-budgets/ibtracs."""
    if os.environ.get("VKE_IBTRACS_DIR"):
        return Path(os.environ["VKE_IBTRACS_DIR"])
    repo = Path(__file__).resolve().parents[3]
    if (repo / "pyproject.toml").exists():
        return repo / "data" / "ibtracs"
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "vke-budgets" / "ibtracs"


def ensure_tracks(directory: str | os.PathLike | None = None,
                  log: Callable[[str], None] | None = None) -> Path:
    """Path of the IBTrACS file in `directory` (default_directory() by
    default), downloaded first if it is missing. The download goes to a
    temporary file that is renamed when complete, so a partial file is never
    taken for the real one. log: function taking a message; by default,
    print."""
    directory = Path(directory) if directory is not None else default_directory()
    path = directory / FILENAME
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    (log or _print)(f"IBTrACS tracks not found in {directory}; downloading {URL}")
    partial = path.with_name(f"{FILENAME}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(URL, partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def open_tracks(path: str | os.PathLike | None = None, start: str | None = None,
                end: str | None = None) -> xr.Dataset:
    """The tropical cyclone points of IBTrACS (see the module docstring), as a
    table on the dimension `point`: sid (storm id), time, lat, lon.

    Args:
        path: the IBTrACS netCDF file; by default ensure_tracks().
        start, end: keep points with start <= time <= end (any string
            numpy.datetime64 reads, e.g. "2010-01-01" or "2010-01-31T21").
    """
    path = path if path is not None else ensure_tracks()
    with xr.open_dataset(path) as ds:
        main = ds["track_type"].values == b"main"
        ds = ds[["sid", "time", "lat", "lon", "nature"]].isel(storm=main).load()
        version = ds.attrs.get("product_version", VERSION)
    sid = np.broadcast_to(ds["sid"].values[:, None], ds["time"].shape)
    # IBTrACS stores the times as floats, a few microseconds off the minute.
    time = ds["time"].dt.round("min").values
    keep = (ds["nature"].values == TROPICAL) & ~np.isnat(time) & np.isfinite(ds["lat"].values)
    if start is not None:
        keep &= time >= np.datetime64(start)
    if end is not None:
        keep &= time <= np.datetime64(end)
    lon = (ds["lon"].values[keep] + 180.0) % 360.0 - 180.0
    return xr.Dataset(
        {"sid": ("point", sid[keep].astype(str)),
         "time": ("point", time[keep]),
         "lat": ("point", ds["lat"].values[keep].astype(float), {"units": "degrees_north"}),
         "lon": ("point", lon.astype(float), {"units": "degrees_east"})},
        attrs={"source": f"IBTrACS {version} ({Path(path).name}), nature TS, main tracks"})


def _print(message: str) -> None:
    print(message, flush=True)
