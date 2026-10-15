"""
The MERRA-2 adapter against the raw files, read independently with netCDF4.
Needs the 2023-01-04 files in $GROUP_SCRATCH/Aidi/VKE_budget/case2023/data;
skipped without them.
"""
import os

import netCDF4
import numpy as np
import pytest

from vke_budgets.sources import merra2
from vke_budgets.standard import check_input, physics_processes

ROOT = os.path.expandvars("$GROUP_SCRATCH/Aidi/VKE_budget/case2023/data")
DATE = "20230104"
FILL = 1e15

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "2023")),
                                reason="MERRA-2 test data not present")


@pytest.fixture(scope="module")
def paths():
    return merra2.find_files(DATE, ROOT)


@pytest.fixture(scope="module")
def ds(paths):
    return merra2.open_day(paths=paths)


def raw(path, name, index):
    """A slab of the raw variable, fill values as NaN, no xarray involved."""
    with netCDF4.Dataset(path) as f:
        f.set_auto_mask(False)
        values = np.asarray(f[name][index])
    # Compare in the file's float32: float64(float32(1e15)) != 1e15.
    return np.where(values == np.float32(FILL), np.nan, values.astype(np.float64))


def test_standard_and_complete(ds):
    check_input(ds)
    assert physics_processes(ds) == (("moist", "turb", "gwd", "dyn", "ana"),
                                     ("moist", "turb", "chem", "dyn", "ana"))
    assert ds.sizes == {"time": 8, "time_inst": 9, "lev": 24, "mlev": 72,
                        "lat": 361, "lon": 576}


def test_times(ds):
    assert str(ds.time_inst.values[0])[:16] == "2023-01-04T00:00"
    assert str(ds.time_inst.values[-1])[:16] == "2023-01-05T00:00"
    assert str(ds.time.values[0])[:16] == "2023-01-04T01:30"


def test_pressure_levels(ds):
    expected = 100.0 * np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
                                 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150])
    np.testing.assert_array_equal(ds.lev, expected)


def test_instantaneous_fields_and_below_ground(ds, paths):
    # 1000-150 hPa are the first 24 of the file's 42 levels.
    for t, path, t_raw in [(2, paths["inst"], 2), (8, paths["inst_next"], 0)]:
        expected = raw(path, "QV", (t_raw, slice(0, 24)))
        ours = ds.q_inst.isel(time_inst=t).values
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(expected))
        np.testing.assert_array_equal(ours[~np.isnan(ours)], expected[~np.isnan(expected)])
    # Over the Tibetan Plateau (31N, 90E) 1000 to 600 hPa are below ground.
    column = ds.q_inst.isel(time_inst=0).sel(lat=31.0, lon=90.0).values
    assert np.isnan(column[:15]).all() and np.isfinite(column[16:]).all()


def test_model_levels_bottom_up(ds, paths):
    t = 5
    p = ds.p.isel(time=t).values
    assert np.all(np.diff(p, axis=0) < 0)                  # pressure falls upward
    np.testing.assert_array_equal(p, raw(paths["mean"], "PL", t)[::-1])
    np.testing.assert_allclose(ds.phi.isel(time=t).values,
                               9.81 * raw(paths["mean"], "H", t)[::-1], rtol=1e-6)
    assert not np.isnan(p).any()                           # never below ground


def test_physics_tendencies(ds, paths):
    t = 1
    for ours, path, name in [("dudt_turb", paths["udt"], "DUDTTRB"),
                             ("dvdt_gwd", paths["udt"], "DVDTGWD"),
                             ("dqdt_moist", paths["qdt"], "DQVDTMST")]:
        expected = raw(path, name, (t, slice(0, 24)))
        np.testing.assert_array_equal(ds[ours].isel(time=t).values, expected)


def test_find_files_rejects_missing_day():
    with pytest.raises(FileNotFoundError):
        merra2.find_files("19790101", ROOT)
