"""AR detection with TempestExtremes (skipped without it)."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vke_budgets import budget, detection
from vke_budgets.sources import ibtracs, merra2

needs_tempest = pytest.mark.skipif(shutil.which("DetectBlobs") is None,
                                   reason="TempestExtremes (DetectBlobs) not installed")
LAT = np.linspace(-90.0, 90.0, 361)
LON = -180.0 + 0.625 * np.arange(576)


@needs_tempest
def test_a_narrow_ridge_is_an_ar_and_the_tropics_are_not():
    """A narrow, long ridge of column KEV at 40 N, and the same at the
    equator: only the first is detected."""
    lat, lon = LAT[:, None], LON[None, :]
    ridge = lambda y0: 30000.0 * np.exp(-((lat - y0) / 2.0) ** 2) * np.exp(-((lon + 150) / 20.0) ** 2)
    kev = xr.DataArray((ridge(40.0) + ridge(0.0))[None], dims=("time_inst", "lat", "lon"),
                       coords={"time_inst": pd.to_datetime(["2023-01-04"]), "lat": LAT, "lon": LON})
    mask = detection.detect(xr.Dataset({"kev": kev}))["mask_kev"].isel(time_inst=0)
    assert mask.sel(lat=40.0, lon=-150.0) and mask.sum() >= detection.MIN_POINTS
    assert not mask.sel(lat=slice(-20, 20)).any()


def tracks(*points):
    """A track table from (time, lat, lon) points."""
    times, lats, lons = zip(*points)
    return xr.Dataset({"time": ("point", pd.to_datetime(list(times)).values),
                       "lat": ("point", np.array(lats, float)),
                       "lon": ("point", np.array(lons, float))})


def great_circle(lat0, lon0):
    """Distance in degrees from (lat0, lon0) to every point of LAT x LON, by
    the spherical law of cosines (a different formula from tc_free's)."""
    phi, phi0 = np.radians(LAT)[:, None], np.radians(lat0)
    cos_d = (np.sin(phi) * np.sin(phi0)
             + np.cos(phi) * np.cos(phi0) * np.cos(np.radians(LON[None, :] - lon0)))
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))


def test_tc_free_is_a_disk_of_4_degrees_at_the_tc_times_only():
    times = pd.to_datetime(["2023-01-04T00", "2023-01-04T03", "2023-01-04T06"]).values
    tc = tracks(("2023-01-04T00", 25.0, 179.375),    # across the dateline
                ("2023-01-04T03", 60.0, -100.0),
                ("2023-01-04T03", 61.0, -99.0),      # two storms, overlapping disks
                ("2023-01-04T04", 30.0, 0.0))        # not a grid time: ignored
    free = detection.tc_free(times, LAT, LON, tc)
    assert free.dims == ("time_inst", "lat", "lon") and free.dtype == bool
    d0, d1, d2 = great_circle(25.0, 179.375), great_circle(60.0, -100.0), great_circle(61.0, -99.0)
    clear = np.abs(d0 - 4.0) > 1e-6                  # away from ties at exactly 4 degrees
    np.testing.assert_array_equal(free[0].values[clear], (d0 > 4.0)[clear])
    clear = (np.abs(d1 - 4.0) > 1e-6) & (np.abs(d2 - 4.0) > 1e-6)
    np.testing.assert_array_equal(free[1].values[clear], ((d1 > 4.0) & (d2 > 4.0))[clear])
    assert free[2].all()
    assert not free[0].sel(lat=29.0, lon=179.375)    # exactly 4 degrees due north: inside
    # the disk reaches both sides of the seam, and is wider in longitude at 60 N
    assert not free[0].sel(lat=25.0, lon=-178.125) and not free[0].sel(lat=25.0, lon=176.25)
    assert not free[1].sel(lat=60.0, lon=-107.5) and free[1].sel(lat=55.5, lon=-100.0)


@needs_tempest
def test_tc_cores_are_cut_before_the_size_filter():
    """A long ridge at 40 N loses only the 4-degree disk around a TC on it;
    a short ridge that the cut leaves below 50 points disappears; and tracks
    without a TC at this time change nothing."""
    lat, lon = LAT[:, None], LON[None, :]
    ridge = lambda y0, x0, width: (30000.0 * np.exp(-((lat - y0) / 2.0) ** 2)
                                   * np.exp(-((lon - x0) / width) ** 2))
    kev = xr.DataArray((ridge(40.0, -150.0, 20.0) + ridge(-40.0, 60.0, 4.0))[None],
                       dims=("time_inst", "lat", "lon"),
                       coords={"time_inst": pd.to_datetime(["2023-01-04"]), "lat": LAT, "lon": LON})
    columns = xr.Dataset({"kev": kev})
    plain = detection.detect(columns)["mask_kev"].isel(time_inst=0)
    short = plain.sel(lat=slice(-60, -20))
    assert detection.MIN_POINTS <= short.sum() < detection.MIN_POINTS + 150

    tc = tracks(("2023-01-04", 40.0, -150.0), ("2023-01-04", -40.0, 60.0))
    cut = detection.detect(columns, tracks=tc)["mask_kev"].isel(time_inst=0)
    core = (great_circle(40.0, -150.0) <= 4.0) | (great_circle(-40.0, 60.0) <= 4.0)
    assert not cut.values[core].any() and not cut.sel(lat=slice(-60, -20)).any()
    north = slice(20, 60)
    np.testing.assert_array_equal(cut.sel(lat=north), (plain & ~core).sel(lat=north))
    assert "tropical cyclones" in cut.attrs["long_name"]

    elsewhere = detection.detect(columns, tracks=tc.assign(time=tc["time"] + np.timedelta64(3, "h")))
    np.testing.assert_array_equal(elsewhere["mask_kev"].isel(time_inst=0), plain)


def test_open_tracks_keeps_tropical_points_of_main_tracks(tmp_path):
    """A small file in the IBTrACS layout: only TS points of main tracks, in
    the time range, with times rounded to the minute and lon in [-180, 180)."""
    t0 = np.datetime64("2010-01-01T00:00:00.000040448", "ns")
    time = t0 + np.arange(4) * np.timedelta64(3, "h")
    path = tmp_path / ibtracs.FILENAME
    xr.Dataset({
        "sid": ("storm", np.array([b"A", b"B"])),
        "track_type": ("storm", np.array([b"main", b"spur-other"])),
        "time": (("storm", "date_time"), np.stack([time, time])),
        "lat": (("storm", "date_time"), np.array([[10, 20, 30, np.nan], [10, 20, 30, 40]], "f4")),
        "lon": (("storm", "date_time"), np.array([[190, 200, 210, 0], [0, 0, 0, 0]], "f4")),
        "nature": (("storm", "date_time"), np.array([[b"TS", b"TS", b"ET", b""]] * 2)),
    }, attrs={"product_version": "v04r01"}).to_netcdf(path)
    points = ibtracs.open_tracks(path, start="2010-01-01T03")
    np.testing.assert_array_equal(points["sid"], ["A"])
    np.testing.assert_array_equal(points["time"], [np.datetime64("2010-01-01T03", "ns")])
    np.testing.assert_allclose(points["lon"], [-160.0])
    assert ibtracs.ensure_tracks(tmp_path) == path        # present: no download


def test_window_mask_is_an_ar_at_either_end():
    t_inst = pd.date_range("2023-01-04", periods=4, freq="3h").values
    mask = xr.DataArray(np.array([0, 1, 0, 0], bool)[:, None], dims=("time_inst", "x"),
                        coords={"time_inst": t_inst})
    windows = detection.window_mask(mask, t_inst[:-1] + np.timedelta64(90, "m"))
    np.testing.assert_array_equal(windows.values[:, 0], [True, True, False])


ROOT = os.environ.get("VKE_TEST_MERRA2_ROOT", "")
MASKS = os.environ.get("VKE_TEST_AR_MASKS", "")
# the reference's thresholds (threshold_v2.txt), with which its masks were made
REFERENCE_THRESHOLDS = {"ivt": -41023.514894, "kev": -478203.362115, "vte": -2105.315636}


@needs_tempest
@pytest.mark.skipif(not (os.path.isdir(ROOT) and os.path.isdir(MASKS)),
                    reason="set VKE_TEST_MERRA2_ROOT and VKE_TEST_AR_MASKS to run")
def test_masks_match_the_reference():
    """2023-01-04, with the reference's thresholds: the same masks as the
    reference, point for point."""
    day = merra2.open_day("20230104", ROOT)
    columns = budget.instantaneous_columns(day[budget.INST_INPUTS].isel(time_inst=slice(0, 8)).load())
    masks = detection.detect(columns, REFERENCE_THRESHOLDS)
    for name, var in (("ivt", "IVT"), ("kev", "IKEV"), ("vte", "IVTE")):
        with xr.open_dataset(os.path.join(MASKS, f"detect_AR_{var}_Lap_v2_20230104.nc")) as ref:
            np.testing.assert_array_equal(masks[f"mask_{name}"].values,
                                          ref["binary_tag"].values.astype(bool))
