"""
Horizontal derivatives on a latitude-longitude grid.

All operators take and return xarray DataArrays with latitude and longitude
dimensions (named `lat` and `lon` by default) in degrees; any other
dimensions (level, time) are carried along. Results are float64 and in SI
units per metre.

Exact relations, with lam = longitude and phi = latitude in radians (see
geometry.py for the radii M and N):

    df/dx = 1 / (N cos phi) * df/dlam                                  (1)
    df/dy = 1 / M * df/dphi = df/dS,  S = northward distance           (2)
    div(u, v) = du/dx + dv/dy - v tan(phi) / N                         (3)

The last term of (3) is not a grid-spacing effect: the eastward unit vector
turns as you move north, so a purely northward flow converges toward the
poles even when v is uniform.

Discretisation, the same as metpy's (and therefore the reference's): a
three-point second-order finite difference on the true, possibly uneven,
spacing, and a one-sided three-point second-order difference at the edges
of the grid. Longitude wraps around on a global grid (see
HorizontalGeometry.periodic_lon). A missing value (NaN) makes every point
whose stencil touches it missing; that is how below-ground points on
pressure levels are handled.
"""
import numpy as np
import xarray as xr

from .geometry import DEFAULT, HorizontalGeometry


def _three_point_derivative(f, s, axis):
    """df/ds along `axis`, for coordinate values s (1D, possibly uneven).

    With spacings h0 = s[i] - s[i-1] and h1 = s[i+1] - s[i], the interior
    formula is exact for quadratics:

        df/ds[i] = -h1 / (h0 (h0+h1)) f[i-1] + (h1-h0) / (h0 h1) f[i]
                   + h0 / (h1 (h0+h1)) f[i+1]

    It reduces to (f[i+1] - f[i-1]) / (2h) on an even grid. The end points
    use the one-sided three-point formulas that are also exact for quadratics.
    """
    f = np.moveaxis(f, axis, -1)
    h = np.diff(s)
    h0, h1 = h[:-1], h[1:]
    out = np.empty_like(f)
    out[..., 1:-1] = (-h1 / (h0 * (h0 + h1)) * f[..., :-2]
                      + (h1 - h0) / (h0 * h1) * f[..., 1:-1]
                      + h0 / (h1 * (h0 + h1)) * f[..., 2:])
    h0, h1 = h[0], h[1]
    out[..., 0] = (-(2 * h0 + h1) / (h0 * (h0 + h1)) * f[..., 0]
                   + (h0 + h1) / (h0 * h1) * f[..., 1]
                   - h0 / (h1 * (h0 + h1)) * f[..., 2])
    h0, h1 = h[-2], h[-1]
    out[..., -1] = (h1 / (h0 * (h0 + h1)) * f[..., -3]
                    - (h0 + h1) / (h0 * h1) * f[..., -2]
                    + (h0 + 2 * h1) / (h1 * (h0 + h1)) * f[..., -1])
    return np.moveaxis(out, -1, axis)


def _periodic_derivative(f, s, axis, period):
    """df/ds along a periodic axis: pad one point on each side by wrapping
    around, take the three-point interior formula, drop the padding."""
    f = np.moveaxis(f, axis, -1)
    f_ext = np.concatenate([f[..., -1:], f, f[..., :1]], axis=-1)
    s_ext = np.concatenate([[s[-1] - period], s, [s[0] + period]])
    out = _three_point_derivative(f_ext, s_ext, -1)[..., 1:-1]
    return np.moveaxis(out, -1, axis)


def is_global_lon(lon_deg, tol=1e-6):
    """True if the longitudes are evenly spaced and cover 360 degrees, so
    that the point after the last one is the first one."""
    step = np.diff(lon_deg)
    return bool(np.allclose(step, step[0], atol=tol)
                and abs(step[0] * lon_deg.size - 360.0) < tol)


def _values(f, dims):
    """float64 ndarray of f and the axis numbers of `dims`."""
    return np.asarray(f, dtype=np.float64), [f.get_axis_num(d) for d in dims]


def _broadcast_lat(values, f, lat):
    """Reshape a 1D array along `lat` so it broadcasts against f."""
    shape = [1] * f.ndim
    shape[f.get_axis_num(lat)] = values.size
    return values.reshape(shape)


def _like(values, f):
    return xr.DataArray(values, dims=f.dims, coords=f.coords)


def ddx(f, geometry: HorizontalGeometry = DEFAULT, lat="lat", lon="lon"):
    """Eastward derivative df/dx (f per m), equation (1).

    Undefined at the poles, where the latitude circle has zero length:
    those rows are NaN.
    """
    lat_deg, lon_deg = f[lat].values, f[lon].values
    fv, (lon_axis,) = _values(f, [lon])
    lam = np.deg2rad(lon_deg)

    periodic = geometry.periodic_lon
    if periodic is None:
        periodic = is_global_lon(lon_deg)
    elif periodic and not is_global_lon(lon_deg):
        raise ValueError("periodic_lon=True needs an evenly spaced global longitude grid")

    if periodic:
        dfdlam = _periodic_derivative(fv, lam, lon_axis, 2 * np.pi)
    else:
        dfdlam = _three_point_derivative(fv, lam, lon_axis)

    circle_radius = geometry.earth.prime_vertical_radius(lat_deg) * np.cos(np.deg2rad(lat_deg))
    circle_radius[np.isclose(np.abs(lat_deg), 90.0)] = np.nan   # poles
    return _like(dfdlam / _broadcast_lat(circle_radius, f, lat), f)


def ddy(f, geometry: HorizontalGeometry = DEFAULT, lat="lat"):
    """Northward derivative df/dy (f per m), equation (2), differenced on
    the true northward distance S(phi). Works for ascending or descending
    latitude."""
    lat_deg = f[lat].values
    fv, (lat_axis,) = _values(f, [lat])
    out = _three_point_derivative(fv, geometry.earth.meridian_distance(lat_deg), lat_axis)
    if geometry.metpy_dy_factor:
        # Reference flaw 1: metpy applies the scale factor a/M to a spacing
        # that is already the true one.
        a_over_m = geometry.earth.a / geometry.earth.meridional_radius(lat_deg)
        out = out * _broadcast_lat(a_over_m, f, lat)
    return _like(out, f)


def gradient(f, geometry: HorizontalGeometry = DEFAULT, lat="lat", lon="lon"):
    """Horizontal gradient (df/dx, df/dy) of a scalar, in f per m."""
    return ddx(f, geometry, lat, lon), ddy(f, geometry, lat)


def divergence(u, v, geometry: HorizontalGeometry = DEFAULT, lat="lat", lon="lon"):
    """Horizontal divergence of the vector (u eastward, v northward),
    equation (3), in units of u per m."""
    lat_deg = v[lat].values
    if geometry.metric_radius is None:
        radius = geometry.earth.prime_vertical_radius(lat_deg)
    else:
        radius = np.full(lat_deg.shape, geometry.metric_radius)   # reference flaw 2
    metric = _broadcast_lat(np.tan(np.deg2rad(lat_deg)) / radius, v, lat)
    return ddx(u, geometry, lat, lon) + ddy(v, geometry, lat) - v * metric
