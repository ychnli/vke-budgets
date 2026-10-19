"""
Atmospheric river detection, as in Ong & Yang (2024) and the reference.

An AR is a blob where the Laplacian of a column field (IVT, the column KEV
or the column VTE) is strongly negative, i.e. a narrow ridge of the field:

    Laplacian(field) < threshold          at a grid point,
    at least 50 connected such points     in a blob,
    |latitude| >= 20 degrees              (no tropics).

The Laplacian is TempestExtremes' _LAPLACIAN{9,7.0}: the field's average
over 9 points on a circle of 7 degrees great-circle radius, minus its value
at the centre, scaled to a Laplacian (units of the field per rad^2). The
detection itself is TempestExtremes' DetectBlobs, run exactly as the
reference runs it (case2023/run_port.sbatch). TempestExtremes must be
installed (conda package tempest-extremes).

The thresholds were chosen so that each field gives about the same global
AR frequency (Zhang et al. 2026, Text S1); REFERENCE_THRESHOLDS are the
reference's (threshold_v2.txt).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import xarray as xr

# Laplacian thresholds of the reference (threshold_v2.txt), in the field's
# units per rad^2.
REFERENCE_THRESHOLDS = {"ivt": -41023.514894, "kev": -478203.362115, "vte": -2105.315636}
MIN_POINTS = 50          # smallest blob, in grid points
MIN_ABS_LAT = 20.0       # degrees


def detect(columns, thresholds=REFERENCE_THRESHOLDS, min_points=MIN_POINTS,
           min_abs_lat=MIN_ABS_LAT, executable="DetectBlobs"):
    """AR masks from instantaneous column fields.

    Args:
        columns: Dataset with any of ivt, kev, vte, on dims (time_inst, lat,
            lon), e.g. a budget result (budget.compute or io.open_budget).
        thresholds: Laplacian threshold for each field to detect from.
    Returns:
        Dataset of boolean masks mask_<field> on (time_inst, lat, lon): True
        inside an AR.
    """
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"{executable} not found: install TempestExtremes "
                                "(conda install -c conda-forge tempest-extremes)")
    fields = [name for name in thresholds if name in columns]
    data = columns[fields].rename(time_inst="time").astype("float32")
    data["lat"].attrs.update(units="degrees_north")
    data["lon"].attrs.update(units="degrees_east")

    masks = {}
    with tempfile.TemporaryDirectory() as tmp:
        infile = os.path.join(tmp, "columns.nc")
        data.to_netcdf(infile)
        for name in fields:
            outfile = os.path.join(tmp, f"mask_{name}.nc")
            laplacian = f"_LAPLACIAN{{9,7.0}}({name}),<,{thresholds[name]}"
            run = subprocess.run(
                [executable, "--in_data", infile, "--out", outfile,
                 "--thresholdcmd", f"{laplacian},0.0",
                 "--filtercmd", f"{laplacian},{min_points}",
                 "--minabslat", str(min_abs_lat)],
                capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"{executable} failed for {name}:\n{run.stdout}\n{run.stderr}")
            with xr.open_dataset(outfile) as out:
                tag = out["binary_tag"].load()
            masks[f"mask_{name}"] = xr.DataArray(
                tag.values.astype(bool), dims=("time_inst", "lat", "lon"),
                coords={"time_inst": columns["time_inst"].values, "lat": columns["lat"].values,
                        "lon": columns["lon"].values},
                attrs={"long_name": f"AR mask from the Laplacian of {name} < {thresholds[name]}",
                       "units": "1"})
    return xr.Dataset(masks)


def window_mask(mask, times):
    """An instantaneous mask on the averaging windows centred at `times`: the
    mean of the masks at the two ends of each window, >= 0.5, i.e. an AR at
    either end. This is the reference's convention (make_website_budget.py).
    Windows whose ends are not both in `mask` are False."""
    half = (np.diff(mask["time_inst"].values).min() / 2) if mask.sizes["time_inst"] > 1 else None
    times = np.asarray(times)
    start = mask.reindex(time_inst=times - half, fill_value=False).values
    end = mask.reindex(time_inst=times + half, fill_value=False).values
    return xr.DataArray(start | end, dims=("time",) + mask.dims[1:],
                        coords={"time": times, **{d: mask[d] for d in mask.dims[1:]}},
                        attrs=mask.attrs)
