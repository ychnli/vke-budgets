"""
Horizontal operators on analytic fields with exact derivatives, for both
Earth models. The default geometry must get them right; the reference's
flaws are tested in test_reference_operator_flaws.py and reproduced in
test_reference_compat.py.
"""
import numpy as np
import pytest
import xarray as xr

import exact
from vke_budgets.geometry import MERRA2_SPHERE, WGS84, HorizontalGeometry
from vke_budgets.horizontal import ddx, ddy, divergence, gradient, is_global_lon

LAT = exact.LAT_MERRA2
LON = exact.LON_MERRA2
H_LON = np.deg2rad(0.625)
INNER_LAT = np.abs(LAT) <= 87.0

EARTHS = {"sphere": MERRA2_SPHERE, "wgs84": WGS84}


def field(values, lat=LAT, lon=LON):
    values = np.broadcast_to(values, (lat.size, lon.size))
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def radii(earth, lat=LAT):
    return (earth.meridional_radius(lat)[:, None], earth.prime_vertical_radius(lat)[:, None],
            np.deg2rad(lat)[:, None])


@pytest.fixture(params=list(EARTHS))
def earth(request):
    return EARTHS[request.param]


def test_ddy_of_northward_distance_is_one(earth):
    """f = S(phi), linear in the northward distance: d f/dy = 1 to rounding,
    including at the poles."""
    geom = HorizontalGeometry(earth=earth)
    out = ddy(field(earth.meridian_distance(LAT)[:, None]), geom)
    np.testing.assert_allclose(out, 1.0, rtol=1e-10)


def test_ddy_of_sin_lat(earth):
    """d sin(phi)/dy = cos(phi)/M, up to truncation (dphi)^2/6 ~ 1.3e-5."""
    m, _, phi = radii(earth)
    out = ddy(field(np.sin(phi)), HorizontalGeometry(earth=earth))
    exact_ = np.broadcast_to(np.cos(phi) / m, out.shape)
    np.testing.assert_allclose(out[INNER_LAT], exact_[INNER_LAT], rtol=2e-5)


def test_ddx_of_eastward_distance_is_one(earth):
    """f = N cos(phi) * lambda: d f/dx = 1 away from the seam (where f
    jumps) and NaN at the poles."""
    _, n, phi = radii(earth)
    lam = np.deg2rad(LON)[None, :]
    out = ddx(field(n * np.cos(phi) * lam), HorizontalGeometry(earth=earth))
    poles = np.isclose(np.abs(LAT), 90.0)
    np.testing.assert_allclose(out[~poles, 1:-1], 1.0, rtol=1e-10)
    assert np.isnan(out[poles]).all()


@pytest.mark.parametrize("k", [1, 6])
def test_ddx_periodic_seam_is_like_any_other_column(earth, k):
    """f = sin(k lambda) on the global grid. With wrap-around the seam
    columns have the same centred-difference error as the rest,
    -(k h)^2/6 relative, instead of twice that."""
    _, n, phi = radii(earth)
    lam = np.deg2rad(LON)[None, :]
    out = ddx(field(np.sin(k * lam)), HorizontalGeometry(earth=earth))
    exact_ = k * np.cos(k * lam) / (n * np.cos(phi))
    err = ((out - exact_) * n * np.cos(phi) / k)[INNER_LAT]
    expected = -(k * H_LON) ** 2 / 6 * np.cos(k * lam)   # leading truncation
    np.testing.assert_allclose(err, np.broadcast_to(expected, err.shape),
                               atol=1e-3 * (k * H_LON) ** 2)


def test_divergence_of_nondivergent_meridional_flow(earth):
    """u = 0, v = V0 a / (N cos phi): v N cos(phi) is constant, so the
    divergence is exactly 0. Fine grid, so that truncation is < 3e-6 of the
    two terms that cancel (see test_reference_operator_flaws.py)."""
    lat = np.linspace(-60.0, 60.0, 2401)
    lon = np.arange(0.0, 360.0, 22.5)
    _, n, phi = radii(earth, lat)
    v = field(10.0 * earth.a / (n * np.cos(phi)), lat, lon)
    div = divergence(xr.zeros_like(v), v, HorizontalGeometry(earth=earth))
    scale = np.abs(v * np.tan(phi) / n)
    rel = (div / scale)[1:-1][np.abs(lat[1:-1]) >= 1.0]   # skip 0/0 at the equator
    np.testing.assert_allclose(rel, 0.0, atol=1e-5)


def test_divergence_of_sin_lat_flow_on_sphere():
    """u = 0, v = sin(phi) on a sphere: div = 1/(R cos phi) d(v cos phi)/dphi
    = cos(2 phi) / (R cos phi). Checks the sign of the metric term."""
    r = MERRA2_SPHERE.a
    phi = np.deg2rad(LAT)[:, None]
    v = field(np.sin(phi))
    div = divergence(xr.zeros_like(v), v)
    exact_ = np.broadcast_to(np.cos(2 * phi) / (r * np.cos(phi)), div.shape)
    np.testing.assert_allclose(div[INNER_LAT], exact_[INNER_LAT],
                               atol=2e-5 / r)


def test_divergence_of_solid_body_rotation_is_zero(earth):
    """u = U cos(phi), v = 0: zonally uniform, so 0 up to the rounding of
    the longitude spacing (~1e-20 against terms of size U/a ~ 3e-6)."""
    phi = np.deg2rad(LAT)[:, None]
    u = field(20.0 * np.cos(phi))
    div = divergence(u, xr.zeros_like(u), HorizontalGeometry(earth=earth))
    np.testing.assert_allclose(div[1:-1], 0.0, atol=1e-18)


def test_descending_latitude_gives_the_same_answer():
    """ERA5 stores latitude north to south."""
    rng = np.random.default_rng(1)
    f = field(rng.standard_normal((LAT.size, LON.size)))
    flipped = f.isel(lat=slice(None, None, -1))
    for a, b in zip(gradient(f), gradient(flipped)):
        np.testing.assert_allclose(a, b.isel(lat=slice(None, None, -1)), rtol=1e-12)


def test_regional_grid_is_not_wrapped():
    lon = np.arange(190.0, 260.0, 0.625)
    assert not is_global_lon(lon)
    assert is_global_lon(LON)
    lam = np.deg2rad(lon)[None, :]
    lat = np.array([0.0, 0.5, 1.0])
    # A quadratic in lambda: the one-sided edge formula is exact for it.
    f = field(lam**2, lat, lon)
    out = ddx(f)
    circle = MERRA2_SPHERE.a * np.cos(np.deg2rad(lat))[:, None]
    np.testing.assert_allclose(out, np.broadcast_to(2 * lam / circle, out.shape), rtol=1e-9)
    with pytest.raises(ValueError):
        ddx(f, HorizontalGeometry(periodic_lon=True))


def test_missing_value_spreads_to_its_stencil_only():
    f = field(np.ones((LAT.size, LON.size))).copy()
    f[100, 200] = np.nan
    gx, gy = gradient(f)
    assert np.isnan(gx[100, 199:202]).all() and np.isfinite(gx[100, [198, 202]]).all()
    assert np.isnan(gy[99:102, 200]).all() and np.isfinite(gy[[98, 102], 200]).all()
    assert np.isfinite(gx[99, 200]) and np.isfinite(gy[100, 199])


def test_extra_dimensions_are_carried_along():
    rng = np.random.default_rng(2)
    f3 = xr.DataArray(rng.standard_normal((3, LAT.size, LON.size)),
                      dims=("lev", "lat", "lon"), coords={"lat": LAT, "lon": LON})
    gx = ddx(f3)
    assert gx.dims == f3.dims
    np.testing.assert_allclose(gx.isel(lev=1), ddx(f3.isel(lev=1)))
    # Dimension order does not matter.
    np.testing.assert_allclose(ddy(f3.transpose("lon", "lev", "lat")).transpose(*f3.dims),
                               ddy(f3))
