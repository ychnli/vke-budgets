"""
Writing and reading budget outputs.

Two layouts.

Clean (the default): one netCDF file per period (normally a month), with
the window terms on `time` and the instantaneous columns (ivt, kev, vte)
on `time_inst`, under the package's names (names.py). A period file holds the instantaneous steps that fall
inside the period, not the one that closes its last window (it opens the
next period), so consecutive files never overlap and open_budget can join
any number of them into one lazy Dataset.

Legacy: the reference's two files per day, analysis_AR_IVKE1_inst<date>.nc
and analysis_AR_IVKE1_tavg<date>.nc, with the reference's variable names
(translated by names.to_legacy), the 1e15 fill value, times as minutes since the first step and levels in hPa,
so that existing downstream scripts keep working.

Precision. Values are stored as float32 with zlib compression. With
`significant_bits`, they are also rounded to that many significant bits
(netCDF BitRound): 12 bits keep about 3.6 decimal digits, a relative error
near 5e-5, and shrink a day of KEV + VTE output from about 200 to 120 MB.
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from . import names

LEGACY_FILL = np.float32(1.0e15)
LEGACY_UNITS = {"kg s-3": "kg / s^3", "kg s-2": "kg / s^2", "kg m-1 s-1": "kg / m / s",
                "m2 s-3": "m^2 / s^3", "m2 s-2": "m^2 / s^2", "kg m-2 s-1": "kg / m^2 / s",
                "s-1": "1 / s", "m3 kg-1": "m^3 / kg"}


def _encoding(ds, significant_bits=None, complevel=1, fill=None):
    """float32, compressed, one lat-lon slab per chunk."""
    enc = {}
    for name, da in ds.data_vars.items():
        chunks = tuple(1 if d not in ("lat", "lon") else da.sizes[d] for d in da.dims)
        e = {"dtype": "float32", "zlib": complevel > 0, "complevel": max(complevel, 1),
             "shuffle": True, "chunksizes": chunks}
        if fill is not None:
            e["_FillValue"] = fill
        if significant_bits is not None:
            e.update(significant_digits=significant_bits, quantize_mode="BitRound")
        enc[name] = e
    return enc


def _write_atomic(ds, path, encoding):
    """Write to a temporary name, then rename, so that an interrupted job
    never leaves a file that looks complete."""
    tmp = f"{path}.tmp"
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


def write_period(result, path, significant_bits=None, complevel=1):
    """Write a budget result in the clean layout. The last instantaneous
    step is dropped when it closes the last window (see the module
    docstring)."""
    if result.sizes["time_inst"] > 0 and result["time_inst"].values[-1] > result["time"].values[-1]:
        result = result.isel(time_inst=slice(0, -1))
    _write_atomic(result, path, _encoding(result, significant_bits, complevel))


def open_budget(paths, **kwargs):
    """Open clean-layout files (a list, or a glob pattern) as one lazy
    Dataset, one dask chunk per time step.

    The files are joined with budget.combine: window terms along `time`,
    instantaneous columns along `time_inst`. (xarray's automatic
    open_mfdataset combine cannot do this: with two time dimensions it
    joins the window terms along the wrong one and fills them with NaN.)
    """
    from .budget import combine

    files = sorted(glob.glob(paths)) if isinstance(paths, str) else list(paths)
    if not files:
        raise FileNotFoundError(f"no files match {paths}")
    parts = [xr.open_dataset(f, chunks={"time": 1, "time_inst": 1}, **kwargs) for f in files]
    out = combine(parts).sortby("time").sortby("time_inst")
    out.attrs = parts[0].attrs
    return out


# --- Legacy layout -------------------------------------------------------

def legacy_paths(out_dir, date):
    return (os.path.join(out_dir, f"analysis_AR_IVKE1_inst{date}.nc"),
            os.path.join(out_dir, f"analysis_AR_IVKE1_tavg{date}.nc"))


def _legacy_dataset(ds, time_dim):
    """Rename time_dim to `time`, encode it as integer minutes, levels in hPa,
    legacy units, MERRA-2-style coordinate attributes."""
    ds = (ds.rename({time_dim: "time"}) if time_dim != "time" else ds).copy()
    # Integer minutes with the units written out, exactly as the reference
    # (xarray's own encoding would write "since 2023-01-04T01:30:00").
    times = pd.DatetimeIndex(ds["time"].values)
    minutes = ((times - times[0]) / pd.Timedelta(minutes=1)).astype(np.int32)
    ds = ds.assign_coords(time=("time", minutes, {
        "long_name": "time", "units": f"minutes since {times[0]:%Y-%m-%d %H:%M:%S}"}))
    ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}
    if "lev" in ds.dims:
        ds = ds.assign_coords(lev=("lev", ds["lev"].values / 100.0,
                                   {"long_name": "vertical level", "units": "hPa"}))
    for da in ds.data_vars.values():
        da.attrs = {**da.attrs,
                    "units": LEGACY_UNITS.get(da.attrs.get("units"), da.attrs.get("units"))}
    return ds


def write_legacy(result, out_dir, date, complevel=0):
    """Write one day of a budget result as the reference's two files, under
    the reference's variable names."""
    result = names.to_legacy(result)
    day = pd.Timestamp(date)
    next_day = day + pd.Timedelta(days=1)
    inst_vars = [v for v in ("IVTE", "IKEV", "IVT") if v in result]
    t_inst = pd.DatetimeIndex(result["time_inst"].values)
    t_win = pd.DatetimeIndex(result["time"].values)
    inst = result[inst_vars].isel(time_inst=np.flatnonzero((t_inst >= day) & (t_inst < next_day)))
    tavg = result.drop_vars(inst_vars).drop_dims("time_inst", errors="ignore")
    tavg = tavg.isel(time=np.flatnonzero((t_win >= day) & (t_win < next_day)))

    inst_path, tavg_path = legacy_paths(out_dir, f"{day:%Y%m%d}")
    for ds, time_dim, path in ((inst, "time_inst", inst_path), (tavg, "time", tavg_path)):
        ds = _legacy_dataset(ds, time_dim)
        enc = _encoding(ds, None, complevel, fill=LEGACY_FILL)
        ds.attrs = {"history": "created by vke_budgets.io.write_legacy", **result.attrs}
        _write_atomic(ds, path, enc)


def open_legacy(out_dir, dates):
    """Read legacy-layout days (a date or a list of YYYYMMDD) into the
    clean layout: the package's names, window terms on `time`, ivt/kev/vte
    on `time_inst`, levels in Pa. Works for the reference's own output files
    too."""
    dates = [dates] if isinstance(dates, str) else list(dates)
    inst, tavg = [], []
    for date in dates:
        inst_path, tavg_path = legacy_paths(out_dir, date)
        inst.append(xr.open_dataset(inst_path).rename(time="time_inst"))
        tavg.append(xr.open_dataset(tavg_path))
    ds = xr.merge([xr.concat(tavg, "time"), xr.concat(inst, "time_inst")],
                  combine_attrs="drop_conflicts")
    if "lev" in ds.dims:
        ds = ds.assign_coords(lev=("lev", ds["lev"].values * 100.0, {"units": "Pa"}))
    return names.from_legacy(ds)
