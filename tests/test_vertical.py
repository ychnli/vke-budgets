"""Vertical operators on fields with exact answers."""
import numpy as np
import pytest
import xarray as xr

from vke_budgets.vertical import (GRAVITY, PressureInterpolator, bottom_value, ddp,
                                  integrate_column)

# MERRA-2 pressure levels from 1000 to 150 hPa, in Pa.
P_NP = 100.0 * np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725,
                         700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150.0])


def column(values, p=P_NP):
    """(lev, x) field on pressure levels; x indexes independent columns."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return xr.DataArray(values, dims=("lev", "x"), coords={"lev": p})


def test_integral_of_one_is_column_mass():
    """<1> = (1000 - 200 hPa) / g; the 150 hPa level is excluded."""
    out = integrate_column(column(np.ones(P_NP.size)))
    np.testing.assert_allclose(out, 800e2 / GRAVITY, rtol=1e-14)


def test_integral_of_linear_profile_is_exact():
    """The trapezoid rule is exact for f linear in p: <p> = (pb^2 - pt^2)/(2g)."""
    out = integrate_column(column(P_NP))
    np.testing.assert_allclose(out, (1000e2**2 - 200e2**2) / (2 * GRAVITY), rtol=1e-14)


def test_integral_starts_at_lowest_level_above_ground():
    """Levels at 1000 and 975 hPa below ground: the integral runs from 950."""
    f = np.ones(P_NP.size)
    f[:2] = np.nan
    np.testing.assert_allclose(integrate_column(column(f)), 750e2 / GRAVITY, rtol=1e-14)


def test_integral_skips_both_layers_around_an_interior_gap():
    f = np.ones(P_NP.size)
    f[5] = np.nan                            # 875 hPa
    np.testing.assert_allclose(integrate_column(column(f)), 750e2 / GRAVITY, rtol=1e-14)


def test_integral_of_all_missing_column_is_nan():
    assert np.isnan(integrate_column(column(np.full(P_NP.size, np.nan)))).all()


def test_integral_does_not_depend_on_level_order():
    rng = np.random.default_rng(0)
    f = column(rng.standard_normal((P_NP.size, 5)))
    np.testing.assert_allclose(integrate_column(f), integrate_column(f.isel(lev=slice(None, None, -1))),
                               rtol=1e-13)


def test_ddp_of_linear_profile_is_exact():
    np.testing.assert_allclose(ddp(column(3.0 * P_NP + 7.0), P_NP), 3.0, rtol=1e-12)


def test_ddp_of_quadratic_on_uneven_levels():
    """f = p^2: the two-layer chord gives p[k+1] + p[k-1] instead of 2 p[k].
    The error is (h1 - h0), zero where the levels are evenly spaced."""
    out = np.asarray(ddp(column(P_NP**2), P_NP))[:, 0]
    np.testing.assert_allclose(out[1:-1], P_NP[2:] + P_NP[:-2], rtol=1e-12)
    even = np.isclose(P_NP[2:] - P_NP[1:-1], P_NP[1:-1] - P_NP[:-2])
    np.testing.assert_allclose(out[1:-1][even], 2 * P_NP[1:-1][even], rtol=1e-12)


def test_ddp_with_pressure_varying_in_space():
    """Model levels: pressure is a field. f = 2p in every column."""
    p = np.stack([P_NP, 0.9 * P_NP], axis=1)
    f = xr.DataArray(2 * p, dims=("lev", "x"))
    np.testing.assert_allclose(ddp(f, xr.DataArray(p, dims=("lev", "x"))), 2.0, rtol=1e-12)


def model_levels(ps):
    """72 sigma-like levels from the top (1 hPa) to just above the surface,
    unevenly spaced, one column per surface pressure in `ps`."""
    s = np.linspace(0.0, 1.0, 72) ** 1.5
    return 100.0 + (np.asarray(ps)[None, :] - 100.0 - 50.0) * s[:, None]


def test_interpolation_of_linear_profile_is_exact():
    ps = np.array([1013e2, 985e2, 800e2])
    p_in = xr.DataArray(model_levels(ps), dims=("lev", "x"))
    f = xr.DataArray(5.0 * p_in.values + 1.0, dims=("lev", "x"))
    out = PressureInterpolator(p_in, P_NP)(f)
    assert out.dims == ("lev", "x")
    np.testing.assert_array_equal(out["lev"], P_NP)
    expected = 5.0 * P_NP[:, None] + 1.0
    inside = P_NP[:, None] <= p_in.values.max(axis=0)[None, :]
    np.testing.assert_allclose(out.values[inside], np.broadcast_to(expected, out.shape)[inside],
                               rtol=1e-12)
    # No extrapolation below the lowest model level.
    assert np.isnan(out.values[~inside]).all()


def test_interpolation_hits_input_levels_exactly():
    p_in = xr.DataArray(P_NP[::-1, None] * np.ones((1, 2)), dims=("lev", "x"))
    f = xr.DataArray(np.random.default_rng(3).standard_normal(p_in.shape), dims=("lev", "x"))
    out = PressureInterpolator(p_in, P_NP)(f)
    np.testing.assert_allclose(out, f.values[::-1], rtol=1e-14)


def test_bottom_value():
    f = np.arange(P_NP.size, dtype=float)[:, None] * np.ones((1, 3))
    f[:2, 1] = np.nan                          # first valid at 950 hPa
    f[:, 2] = np.nan
    out = bottom_value(column(f))
    np.testing.assert_array_equal(out[:2], [0.0, 2.0])
    assert np.isnan(out[2])
    # Same answer for top-down level order.
    np.testing.assert_array_equal(bottom_value(column(f).isel(lev=slice(None, None, -1)))[:2],
                                  [0.0, 2.0])
