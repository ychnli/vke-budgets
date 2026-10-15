"""
With geometry.REFERENCE, our operators must reproduce the reference
operators ($REF/py/tools_comp.py and tools_ncl_equivalent.py), flaws
included, far below float32 rounding. Fields are random, with below-ground
missing values like real pressure-level data.
"""
import numpy as np
import pytest
import xarray as xr

import exact
from vke_budgets.geometry import REFERENCE
from vke_budgets.horizontal import divergence, gradient
from vke_budgets.vertical import GRAVITY, PressureInterpolator, bottom_value, ddp, integrate_column

LAT = exact.LAT_MERRA2
LON = exact.LON_MERRA2
P_NP = 100.0 * np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
                         700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150.0])
RTOL = 1e-10


@pytest.fixture(scope="module")
def ncl_tools(tools_comp):
    import tools_ncl_equivalent
    return tools_ncl_equivalent


def smooth_random(shape, seed):
    """A random field with structure at all scales down to a few grid points."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(shape)
    for axis in (-1, -2):
        f = (np.roll(f, 1, axis) + 2 * f + np.roll(f, -1, axis)) / 4
    return f


def pressure_level_field(seed, n_lev=4):
    """(lev, lat, lon) field, NaN below a random 'ground' in the lowest levels."""
    f = smooth_random((n_lev, LAT.size, LON.size), seed)
    ground = smooth_random((LAT.size, LON.size), seed + 100)
    f[0][ground > 0.3] = np.nan
    f[1][ground > 0.8] = np.nan
    return xr.DataArray(f, dims=("lev", "lat", "lon"),
                        coords={"lat": ("lat", LAT, {"units": "degrees_north"}),
                                "lon": ("lon", LON, {"units": "degrees_east"})})


def as_array(result):
    """Plain ndarray from what the reference returns (pint, xarray or both)."""
    if isinstance(result, xr.DataArray):
        result = result.data
    return np.asarray(getattr(result, "magnitude", result))


def assert_same(ours, ref, rtol=RTOL):
    ours, ref = np.asarray(ours), as_array(ref)
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))
    ok = np.isfinite(ref)
    np.testing.assert_allclose(ours[ok], ref[ok], rtol=rtol, atol=rtol * np.abs(ref[ok]).max())


def test_gradient(tools_comp):
    f = pressure_level_field(0)
    ref_x, ref_y = tools_comp.gradient_2d(f, longitude=f["lon"], latitude=f["lat"])
    ours_x, ours_y = gradient(f, REFERENCE)
    poles = np.isclose(np.abs(LAT), 90.0)
    # The reference returns huge finite values at the poles; we return NaN.
    # The reference output blanks |lat| >= 87 anyway.
    assert_same(ours_x[:, ~poles], as_array(ref_x)[:, ~poles])
    assert_same(ours_y, ref_y)


def test_divergence(tools_comp):
    u, v = pressure_level_field(1), pressure_level_field(2)
    ref = tools_comp.divergence_2d(u, v, longitude=u["lon"], latitude=u["lat"])
    poles = np.isclose(np.abs(LAT), 90.0)
    assert_same(divergence(u, v, REFERENCE)[:, ~poles], as_array(ref)[:, ~poles])


def model_level_pressure(seed):
    """(lev, lat, lon) model-level pressure, bottom-up, varying with 'surface
    pressure' between 600 and 1030 hPa."""
    rng = np.random.default_rng(seed)
    ps = 100.0 * rng.uniform(600, 1030, (8, 9))
    s = np.linspace(1.0, 0.0, 72) ** 1.5
    return xr.DataArray(100.0 + (ps[None] - 100.0 - 60.0) * s[:, None, None],
                        dims=("lev", "lat", "lon"))


def test_ddp(ncl_tools):
    p = model_level_pressure(0)
    f = xr.DataArray(np.random.default_rng(1).standard_normal(p.shape), dims=p.dims)
    assert_same(ddp(f, p), ncl_tools.center_finite_diff_n(f, p, "lev"))


def test_interpolation(ncl_tools):
    p = model_level_pressure(2)
    f = xr.DataArray(np.random.default_rng(3).standard_normal(p.shape), dims=p.dims)
    ref = ncl_tools.linint1_n_apply(f, ncl_tools.linint1_n_weights(p, P_NP, f, "lev"))
    # The reference stores its weights in float32.
    assert_same(PressureInterpolator(p, P_NP)(f), ref, rtol=1e-6)


def test_column_integral(tools_comp):
    f = pressure_level_field(4, n_lev=P_NP.size).assign_coords(lev=P_NP)
    n_int = int((P_NP >= 200e2).sum())
    p_int = xr.DataArray(P_NP[:n_int] / 100.0, dims="lev", attrs={"units": "hPa"})
    ref = tools_comp.integZ(f.isel(lev=slice(0, n_int)).drop_vars("lev"), p_int, GRAVITY)
    assert_same(integrate_column(f), ref)


def test_bottom_value(ncl_tools):
    f = pressure_level_field(5, n_lev=P_NP.size).assign_coords(lev=P_NP)
    ref = ncl_tools.linmsg_n(f, "lev").isel(lev=0)
    assert_same(bottom_value(f), ref)
