"""
Do the reference's horizontal operators have the flaws listed in
VERIFICATION_MERRA2.md §4 (items 7, 8 and 10)? Each test applies the
reference's own gradient_2d / divergence_2d (reference/py/tools_comp.py, metpy
1.6.3) to an analytic field on the MERRA-2 grid, and compares the result with
the exact derivative on the WGS84 ellipsoid (see exact.py).

The tests assert the reference's behaviour as it is, flaws included, so they
pass. They document the flaws and size them; they are not tests of our code.

1. Meridional derivatives are too large by a/M(phi): +0.67% at the equator,
   +0.17% at 45 deg, -0.30% at 80 deg. metpy takes dy as the true meridian arc
   length M dphi, and then multiplies by pyproj's meridional scale factor a/M,
   which converts a *nominal* spacing a dphi into a true one. The ellipsoid is
   counted twice. Zonal derivatives are correct: there dx is nominal (a dlam)
   and the scale factor a/(N cos phi) makes it true.
2. divergence_2d adds the spherical metric term -v tan(phi)/R_mean to those
   ellipsoidal derivatives; on the ellipsoid it is -v tan(phi)/N(phi).
3. Longitude derivatives are not cyclic: at the two seam columns (-180 and
   179.375 deg) metpy uses a one-sided difference and ignores the neighbour
   across the seam.
"""
import numpy as np
import pytest
import xarray as xr

from exact import (A_WGS84, R_MEAN, LAT_MERRA2, LON_MERRA2, meridian_distance,
                   meridional_radius, prime_vertical_radius)

LAT = LAT_MERRA2
LON = LON_MERRA2
PHI = np.deg2rad(LAT)[:, None]          # (lat, 1), radians
LAM = np.deg2rad(LON)[None, :]          # (1, lon), radians
M = meridional_radius(LAT)[:, None]
N = prime_vertical_radius(LAT)[:, None]

# Rows the reference keeps (it blanks poleward of 87 deg), and columns away
# from the +-180 deg seam.
INNER_LAT = np.abs(LAT) <= 87.0
INNER_LON = slice(1, -1)


def as_field(values, lat=LAT, lon=LON):
    """A (lat, lon) DataArray laid out like a MERRA-2 field."""
    values = np.broadcast_to(values, (lat.size, lon.size)).astype("float64")
    return xr.DataArray(
        values, dims=("lat", "lon"),
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})})


def magnitude(result):
    """Plain ndarray, in SI, from what metpy returns."""
    if isinstance(result, xr.DataArray):
        result = result.metpy.unit_array
    return np.asarray(result.m_as("1/m"))


def ref_grad(tools_comp, values):
    """(df/dx, df/dy) as the reference computes them (its grad_xy)."""
    f = as_field(values)
    gx, gy = tools_comp.gradient_2d(f, longitude=f["lon"], latitude=f["lat"])
    return magnitude(gx), magnitude(gy)


def ref_div(tools_comp, u, v, lat=LAT, lon=LON):
    """Horizontal divergence as the reference computes it (its div_xy)."""
    u, v = as_field(u, lat, lon), as_field(v, lat, lon)
    return magnitude(tools_comp.divergence_2d(u, v, longitude=u["lon"],
                                              latitude=u["lat"]))


# ---------------------------------------------------------------------------
# 1. Meridional derivative: too large by a/M
# ---------------------------------------------------------------------------

def test_mechanism_true_dy_times_scale_factor():
    """Where the a/M comes from: metpy's dy is already the true arc length,
    and the meridional scale factor it multiplies by is a/M."""
    from metpy.calc.tools import nominal_lat_lon_grid_deltas
    from metpy.units import units
    from pyproj import CRS, Proj

    crs = CRS("+proj=latlon")
    dx, dy = nominal_lat_lon_grid_deltas(units.Quantity(LON, "deg"),
                                         units.Quantity(LAT, "deg"), crs.get_geod())
    # dx is nominal, a * dlam, the same at every latitude...
    np.testing.assert_allclose(dx.m_as("m"), A_WGS84 * np.deg2rad(np.diff(LON)), rtol=1e-12)
    # ...but dy is the true meridian arc M * dphi, not the nominal a * dphi.
    np.testing.assert_allclose(dy.m_as("m"), np.diff(meridian_distance(LAT)), rtol=1e-9)

    factors = Proj(crs).get_factors(np.zeros_like(LAT), LAT)
    # The scale factors convert nominal spacings to true ones. Right for dx,
    # a second ellipsoid correction for dy.
    np.testing.assert_allclose(factors.meridional_scale, A_WGS84 / M[:, 0], rtol=1e-9)
    inner = np.abs(LAT) < 90
    np.testing.assert_allclose(factors.parallel_scale[inner],
                               A_WGS84 / (N[inner, 0] * np.cos(PHI[inner, 0])), rtol=1e-9)


def test_meridional_derivative_of_northward_distance(tools_comp):
    """f = S(phi), the distance north of the equator. Exactly d f/dy = 1.

    f is linear in the true northward distance, so a finite difference on
    the true spacing is exact: any departure from 1 is the metric, not
    truncation. The reference returns a/M instead."""
    _, gy = ref_grad(tools_comp, meridian_distance(LAT)[:, None])
    expected_ref = np.broadcast_to(A_WGS84 / M, gy.shape)
    np.testing.assert_allclose(gy[INNER_LAT], expected_ref[INNER_LAT], rtol=1e-8)

    rel_err = gy[INNER_LAT] - 1.0
    assert rel_err.max() == pytest.approx(A_WGS84 / M.min() - 1, rel=1e-6)   # equator
    assert 0.0067 < rel_err.max() < 0.0068


def test_meridional_derivative_of_sin_lat(tools_comp):
    """f = sin(phi). Exactly d f/dy = cos(phi) / M. Same a/M factor; the
    only other error is centred-difference truncation, (dphi)^2/6 ~ 1e-5."""
    _, gy = ref_grad(tools_comp, np.sin(PHI))
    exact = np.broadcast_to(np.cos(PHI) / M, gy.shape)
    ratio = gy[INNER_LAT] / exact[INNER_LAT]
    expected_ratio = np.broadcast_to(A_WGS84 / M, gy.shape)[INNER_LAT]
    np.testing.assert_allclose(ratio, expected_ratio, rtol=5e-5)


# ---------------------------------------------------------------------------
# Zonal derivative: correct
# ---------------------------------------------------------------------------

def test_zonal_derivative_of_eastward_distance(tools_comp):
    """f = N cos(phi) * lambda, the distance east of the prime meridian.
    Exactly d f/dx = 1, and the reference gets it (away from the seam,
    where f jumps by the circumference)."""
    gx, _ = ref_grad(tools_comp, N * np.cos(PHI) * LAM)
    np.testing.assert_allclose(gx[INNER_LAT, INNER_LON], 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. Divergence: spherical metric term on ellipsoidal derivatives
# ---------------------------------------------------------------------------

def test_divergence_of_nondivergent_meridional_flow(tools_comp):
    """u = 0, v = V0 a / (N cos phi). Then v N cos(phi) is constant, so the
    exact divergence is 0: dv/dy = +v tan(phi)/N and the metric term
    -v tan(phi)/N cancel it.

    The reference returns (a/M) dv/dy - v tan(phi)/R_mean
                        = (v tan(phi) / N) * [(a/M - 1) - (N/R_mean - 1)].
    The first bracket term is flaw 1, the second is the mixed Earth model.

    sec(phi) is steep, so on the 0.5 deg grid centred-difference truncation
    is ~3e-4 of each term at 60 deg, as large as the flaws. The flaws do not
    depend on the grid, so this test uses a 0.05 deg latitude grid, where
    truncation is below 3e-6."""
    lat = np.linspace(-60.0, 60.0, 2401)
    lon = np.arange(0.0, 360.0, 22.5)     # v is zonally uniform
    phi = np.deg2rad(lat)[:, None]
    m, n = meridional_radius(lat)[:, None], prime_vertical_radius(lat)[:, None]

    v = np.broadcast_to(10.0 * A_WGS84 / (n * np.cos(phi)), (lat.size, lon.size))
    div = ref_div(tools_comp, np.zeros_like(v), v, lat, lon)

    scale = v * np.tan(phi) / n           # size of each of the cancelling terms
    predicted = (A_WGS84 / m - 1) - (n / R_MEAN - 1)
    inner = slice(1, -1)                  # one-sided differences at the ends
    eq = np.abs(lat[inner]) < 1.0         # 0/0 at the equator, where both vanish
    rel = (div / scale)[inner][~eq]
    np.testing.assert_allclose(rel, np.broadcast_to(predicted, div.shape)[inner][~eq],
                               atol=1e-5)

    # The two errors partly cancel; neither is zero.
    at45 = lat == 45.0
    assert abs((div / scale)[at45, 0]) > 5e-4


# ---------------------------------------------------------------------------
# 3. Longitude seam: not cyclic
# ---------------------------------------------------------------------------

def test_seam_neighbour_is_ignored(tools_comp):
    """Perturb the last column (179.375 deg). With a cyclic stencil d/dx at
    the first column (-180 deg) would change; in the reference it does not."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((LAT.size, LON.size))
    g = f.copy()
    g[:, -1] += 1.0
    gx_f, _ = ref_grad(tools_comp, f)
    gx_g, _ = ref_grad(tools_comp, g)
    rows = INNER_LAT
    np.testing.assert_array_equal(gx_f[rows, 0], gx_g[rows, 0])
    # Sanity check: the interior stencil does see its neighbours.
    assert np.all(gx_f[rows, -2] != gx_g[rows, -2])


@pytest.mark.parametrize("k", [1, 6])
def test_seam_truncation_error_doubles(tools_comp, k):
    """f = sin(k lambda), periodic. Exactly d f/dx = k cos(k lambda)/(N cos phi).

    Centred differences err by -h^2 f'''/6; the one-sided second-order
    stencil at the seam errs by +h^2 f'''/3, twice as large and of opposite
    sign. Both are small for smooth fields: the seam is a consistency flaw,
    not a bias."""
    h = np.deg2rad(0.625)
    gx, _ = ref_grad(tools_comp, np.sin(k * LAM))
    exact = k * np.cos(k * LAM) / (N * np.cos(PHI))
    unit = k / (N * np.cos(PHI))          # largest |exact| on each row
    err = ((gx - exact) / unit)[INNER_LAT]

    centred = np.abs(err[:, INNER_LON]).max()
    seam = np.abs(err[:, 0]).max()        # -180 deg, where |f'''| is largest
    np.testing.assert_allclose(centred, (k * h) ** 2 / 6, rtol=1e-2)
    np.testing.assert_allclose(seam, (k * h) ** 2 / 3, rtol=1e-2)
