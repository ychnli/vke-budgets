"""
Atmospheric river detection following the TempestExtremes method

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

For detection in reanalysis, a point is excluded from an AR if it lies within 
4 great circle distance from an ibtracs TC center. 
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

# Laplacian thresholds in the field's units per rad^2. IVT and VTE are Ong &
# Yang's (2024): -40000 is the usual TempestExtremes IVT threshold, and -2000
# for VTE was chosen to give about the same global mean AR frequency. The KEV
# threshold is chosen likewise, to roughly match the AR climatology of the
# other two.
DEFAULT_THRESHOLDS = {"ivt": -40000, "kev": -460000, "vte": -2000}

# smallest blob, in grid points
MIN_POINTS = 50

MIN_ABS_LAT = 20.0

# radius of the excluded TC core, in degrees of great-circle distance
TC_RADIUS = 4.0


def tc_free(times: ArrayLike, lat: ArrayLike, lon: ArrayLike, tracks: xr.Dataset,
            radius: float = TC_RADIUS) -> xr.DataArray:
    """False within `radius` degrees (inclusive) of great-circle distance of a tropical
    cyclone centre, True elsewhere, on (time_inst, lat, lon).

    The distance between a grid point (phi, lambda) and a TC centre
    (phi0, lambda0) is the central angle (haversine formula)

        d = 2 arcsin sqrt( sin^2((phi - phi0)/2)
                           + cos(phi) cos(phi0) sin^2((lambda - lambda0)/2) ).

    Args:
        times, lat, lon: the grid (degrees).
        tracks: TC points with time, lat, lon on one dimension
            (sources.ibtracs.open_tracks). Points at other times than `times`
            are ignored.
    """
    times, lat, lon = np.asarray(times), np.asarray(lat, float), np.asarray(lon, float)
    free = np.ones((times.size, lat.size, lon.size), bool)
    step = {t: i for i, t in enumerate(times)}
    for t, lat0, lon0 in zip(tracks["time"].values, tracks["lat"].values, tracks["lon"].values):
        if t not in step:
            continue
        rows = np.abs(lat - lat0) <= radius          # the only rows the disk can reach
        phi, phi0 = np.radians(lat[rows])[:, None], np.radians(lat0)
        half_dlon = np.radians(lon - lon0)[None, :] / 2
        h = np.sin((phi - phi0) / 2) ** 2 + np.cos(phi) * np.cos(phi0) * np.sin(half_dlon) ** 2
        distance = np.degrees(2 * np.arcsin(np.sqrt(np.minimum(h, 1.0))))
        # A point at exactly `radius` (up to rounding) is inside.
        free[step[t], rows] &= distance > radius + 1e-9
    return xr.DataArray(free, dims=("time_inst", "lat", "lon"),
                        coords={"time_inst": times, "lat": lat, "lon": lon},
                        attrs={"long_name": f"farther than {radius} degrees from a tropical "
                                            "cyclone centre", "units": "1"})


def detect(columns: xr.Dataset, thresholds: dict[str, float] = DEFAULT_THRESHOLDS,
           min_points: int = MIN_POINTS, min_abs_lat: float = MIN_ABS_LAT,
           tracks: xr.Dataset | None = None, tc_radius: float = TC_RADIUS,
           executable: str = "DetectBlobs") -> xr.Dataset:
    """AR masks from instantaneous column fields.

    Args:
        columns: Dataset with any of ivt, kev, vte, on dims (time_inst, lat,
            lon), e.g. a budget result (budget.compute or io.open_budget).
        thresholds: Laplacian threshold for each field to detect from.
        tracks: tropical cyclone points (sources.ibtracs.open_tracks); if
            given, their cores (tc_radius degrees) are excluded before the
            min_points filter. For a reanalysis; None for model output.
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
    extra = excluded = ""
    if tracks is not None:
        free = tc_free(columns["time_inst"].values, columns["lat"].values,
                       columns["lon"].values, tracks, tc_radius)
        data["tc_free"] = (("time", "lat", "lon"), free.values.astype("float32"))
        extra = ";tc_free,>,0.5,0.0"
        excluded = f", outside {tc_radius} degrees of tropical cyclones"

    masks = {}
    with tempfile.TemporaryDirectory() as tmp:
        infile = os.path.join(tmp, "columns.nc")
        data.to_netcdf(infile)
        for name in fields:
            outfile = os.path.join(tmp, f"mask_{name}.nc")
            laplacian = f"_LAPLACIAN{{9,7.0}}({name}),<,{thresholds[name]}"
            run = subprocess.run(
                [executable, "--in_data", infile, "--out", outfile,
                 "--thresholdcmd", f"{laplacian},0.0{extra}",
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
                attrs={"long_name": f"AR mask from the Laplacian of {name} < {thresholds[name]}"
                                    f"{excluded}",
                       "units": "1"})
    return xr.Dataset(masks)


def window_mask(mask: xr.DataArray, times: ArrayLike) -> xr.DataArray:
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
