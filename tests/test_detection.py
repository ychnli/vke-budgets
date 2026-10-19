"""AR detection with TempestExtremes (skipped without it)."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vke_budgets import budget, detection
from vke_budgets.sources import merra2

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


def test_window_mask_is_an_ar_at_either_end():
    t_inst = pd.date_range("2023-01-04", periods=4, freq="3h").values
    mask = xr.DataArray(np.array([0, 1, 0, 0], bool)[:, None], dims=("time_inst", "x"),
                        coords={"time_inst": t_inst})
    windows = detection.window_mask(mask, t_inst[:-1] + np.timedelta64(90, "m"))
    np.testing.assert_array_equal(windows.values[:, 0], [True, True, False])


ROOT = os.environ.get("VKE_TEST_MERRA2_ROOT", "")
MASKS = os.environ.get("VKE_TEST_AR_MASKS", "")


@needs_tempest
@pytest.mark.skipif(not (os.path.isdir(ROOT) and os.path.isdir(MASKS)),
                    reason="set VKE_TEST_MERRA2_ROOT and VKE_TEST_AR_MASKS to run")
def test_masks_match_the_reference():
    """2023-01-04: the same masks as the reference, point for point."""
    day = merra2.open_day("20230104", ROOT)
    columns = budget.instantaneous_columns(day[budget.INST_INPUTS].isel(time_inst=slice(0, 8)).load())
    masks = detection.detect(columns)
    for name, var in (("ivt", "IVT"), ("kev", "IKEV"), ("vte", "IVTE")):
        with xr.open_dataset(os.path.join(MASKS, f"detect_AR_{var}_Lap_v2_20230104.nc")) as ref:
            np.testing.assert_array_equal(masks[f"mask_{name}"].values,
                                          ref["binary_tag"].values.astype(bool))
