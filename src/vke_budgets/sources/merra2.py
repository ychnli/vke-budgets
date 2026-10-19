"""
MERRA-2 adapter: five files per day to the standard input Dataset
(see standard.py).

Collections used, for a date D:

    inst3_3d_asm_Np   instantaneous state on 42 pressure levels at 00, 03,
                      ..., 21Z of D, and of D+1 for its 00Z
    tavg3_3d_asm_Nv   3-hour means on the 72 model levels, centred at 01:30,
                      04:30, ..., 22:30Z. Levels run top-down in the file.
    tavg3_3d_udt_Np   3-hour mean wind tendencies by process, pressure levels
    tavg3_3d_qdt_Np   3-hour mean water vapour tendencies by process

Below-ground points on the Np levels have the fill value 1e15, which
becomes NaN. H is geopotential height, so phi = g H, formed in float64.
Everything else stays float32, as in the files.

The Dataset is lazy (dask, one chunk per time step): opening a day reads
nothing, and each time step is read when it is used.
"""
import glob
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from ..standard import check_input
from ..vertical import GRAVITY

COLLECTIONS = {
    "inst": "inst3_3d_asm_Np",
    "inst_next": "inst3_3d_asm_Np",
    "mean": "tavg3_3d_asm_Nv",
    "udt": "tavg3_3d_udt_Np",
    "qdt": "tavg3_3d_qdt_Np",
}

# MERRA-2 name -> standard name, for the physics tendencies.
PROCESS_CODES = {"MST": "moist", "TRB": "turb", "GWD": "gwd", "CHM": "chem",
                 "DYN": "dyn", "ANA": "ana"}
MOMENTUM_CODES = ("MST", "TRB", "GWD", "DYN", "ANA")
MOISTURE_CODES = ("MST", "TRB", "CHM", "DYN", "ANA")


def find_files(date, root):
    """Paths of the five files for `date` (YYYYMMDD), looked up as
    root/YYYY/MERRA2_*.<collection>.<date>.nc4. The stream number (100, 200,
    300, 400, 401) is matched by the glob, so reprocessed streams are found
    too; more than one match is an error."""
    next_date = (datetime.strptime(date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    paths = {}
    for key, collection in COLLECTIONS.items():
        day = next_date if key == "inst_next" else date
        pattern = os.path.join(root, day[:4], f"MERRA2_*.{collection}.{day}.nc4")
        hits = sorted(glob.glob(pattern))
        if len(hits) != 1:
            raise FileNotFoundError(f"{len(hits)} files match {pattern}; expected one")
        paths[key] = hits[0]
    return paths


def _open(path):
    # One dask chunk per time step. The files' own chunks are much smaller
    # (91 x 144 points), and would make thousands of tiny tasks.
    return xr.open_dataset(path, chunks={"time": 1, "lev": -1, "lat": -1, "lon": -1})


def _attrs(units, long_name):
    return {"units": units, "long_name": long_name}


def open_day(date=None, root=None, paths=None, p_bottom=1000e2, p_top=150e2, g=GRAVITY):
    """One day of MERRA-2 as a standard input Dataset.

    Args:
        date: YYYYMMDD. With `root`, the files are found by find_files.
        root: directory holding one subdirectory per year.
        paths: dict of the five paths (keys as in COLLECTIONS), instead of
            date and root.
        p_bottom, p_top: range of pressure levels kept (Pa). The default,
            1000-150 hPa (24 levels), is the reference's.
        g: gravity (m s-2) for phi = g H. The reference uses 9.81.
    """
    if paths is None:
        paths = find_files(date, root)
    raw = {key: _open(path) for key, path in paths.items()}

    def pressure_levels(da):
        """Keep p_bottom..p_top and label the levels in Pa."""
        pa = da["lev"].values * 100.0
        da = da.isel(lev=np.flatnonzero((pa <= p_bottom) & (pa >= p_top)))
        return da.assign_coords(lev=da["lev"] * 100.0)

    # Instantaneous: today's steps plus 00Z of the next day.
    def inst(name):
        both = [raw["inst"][name], raw["inst_next"][name].isel(time=[0])]
        da = xr.concat(both, dim="time").rename(time="time_inst")
        return pressure_levels(da) if "lev" in da.dims else da

    # Window means on model levels, flipped to bottom-up.
    def mean(name):
        da = raw["mean"][name].isel(lev=slice(None, None, -1)).rename(lev="mlev")
        return da.assign_coords(mlev=da["mlev"].astype(int))

    ds = xr.Dataset({
        "q_inst": inst("QV").assign_attrs(_attrs("kg kg-1", "specific humidity")),
        "u_inst": inst("U").assign_attrs(_attrs("m s-1", "eastward wind")),
        "v_inst": inst("V").assign_attrs(_attrs("m s-1", "northward wind")),
        "ps_inst": inst("PS").assign_attrs(_attrs("Pa", "surface pressure")),
        "p": mean("PL").assign_attrs(_attrs("Pa", "pressure at model levels")),
        "q": mean("QV").assign_attrs(_attrs("kg kg-1", "specific humidity")),
        "u": mean("U").assign_attrs(_attrs("m s-1", "eastward wind")),
        "v": mean("V").assign_attrs(_attrs("m s-1", "northward wind")),
        # In float64: phi ~ 1e5 m2 s-2, so float32 resolves it to ~0.01 m2 s-2
        # while the differences that grad(phi) takes between grid points are
        # 10-100 m2 s-2. An extra float32 rounding moves PEKE by ~1e-5.
        "phi": (g * mean("H").astype(np.float64)).assign_attrs(
            _attrs("m2 s-2", f"geopotential, {g} * H")),
        "omega": mean("OMEGA").assign_attrs(_attrs("Pa s-1", "vertical pressure velocity")),
    })
    for code in MOMENTUM_CODES:
        proc = PROCESS_CODES[code]
        for comp, long in (("u", "eastward"), ("v", "northward")):
            ds[f"d{comp}dt_{proc}"] = pressure_levels(raw["udt"][f"D{comp.upper()}DT{code}"]) \
                .assign_attrs(_attrs("m s-2", f"{long} wind tendency, {proc}"))
    for code in MOISTURE_CODES:
        proc = PROCESS_CODES[code]
        ds[f"dqdt_{proc}"] = pressure_levels(raw["qdt"][f"DQVDT{code}"]) \
            .assign_attrs(_attrs("kg kg-1 s-1", f"specific humidity tendency, {proc}"))

    # Plain coordinate attributes (the files also carry vmax, vmin and
    # valid_range, which say nothing here).
    ds = ds.assign_coords(
        lev=ds["lev"].assign_attrs(_attrs("Pa", "pressure")),
        mlev=ds["mlev"].assign_attrs(_attrs("1", "MERRA-2 model layer, 1 = top")),
        lat=ds["lat"].copy().assign_attrs(_attrs("degrees_north", "latitude")),
        lon=ds["lon"].copy().assign_attrs(_attrs("degrees_east", "longitude")),
    )
    ds.attrs = {"source": "MERRA-2",
                **{f"file_{key}": os.path.basename(path) for key, path in paths.items()}}

    _check_times(ds, raw)
    check_input(ds)
    return ds


def _check_times(ds, raw):
    """The four collections must share one day, and each 3-hour window must
    be bracketed by the instantaneous steps on either side of it."""
    for key in ("udt", "qdt"):
        if not np.array_equal(raw[key]["time"].values, ds["time"].values):
            raise ValueError(f"{key} times differ from the tavg3_3d_asm_Nv times")
    t0, t1 = ds["time_inst"].values[:-1], ds["time_inst"].values[1:]
    if not (np.all(t1 - t0 == np.timedelta64(3, "h"))
            and np.all(ds["time"].values - t0 == np.timedelta64(90, "m"))):
        raise ValueError("instantaneous steps do not bracket the 3-hour windows")
