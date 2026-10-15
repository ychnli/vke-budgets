"""
Vertical operators: column integral, d/dp, interpolation to pressure levels,
and the value at the bottom of the column.

All take and return xarray DataArrays. Pressure is in Pa. A missing value
(NaN) marks a point below the ground. The level order (top-down or
bottom-up) does not matter.
"""
import numpy as np
import xarray as xr

# The reference's g (m s-2), used for every column integral and for
# geopotential = g * height.
GRAVITY = 9.81


def _pressure_along(p, f, dim):
    """p as an ndarray that broadcasts against f: either the 1D coordinate
    along `dim` or a field with the same dims as f (e.g. model-level
    pressure, which varies in space)."""
    if isinstance(p, xr.DataArray) and p.dims != (dim,):
        return np.asarray(p.transpose(*f.dims), dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    shape = [1] * f.ndim
    shape[f.get_axis_num(dim)] = p.size
    return p.reshape(shape)


def integrate_column(f, p=None, dim="lev", g=GRAVITY, p_bottom=1000e2, p_top=200e2):
    """Mass-weighted column integral <f> = (1/g) * integral of f dp from
    p_top to p_bottom, by the trapezoid rule over pressure levels.

        <f> = (1/g) * sum over layers k of (f[k] + f[k+1]) / 2 * |p[k] - p[k+1]|

    Only levels with p_top <= p <= p_bottom take part. A layer is skipped
    when either of its two levels is missing, so over high ground the
    integral starts at the lowest level that is above ground; no partial
    layer down to the surface is added. This is the reference's integZ.
    A column with no complete layer is NaN.

    If f is per unit mass (e.g. m2 s-2), <f> is per unit area (kg s-2).

    Args:
        f: field with a pressure-level dimension `dim`.
        p: 1D pressure (Pa) along `dim`; defaults to f's `dim` coordinate.
        g: gravity (m s-2).
    """
    p = f[dim] if p is None else p
    keep = (np.asarray(p) >= p_top) & (np.asarray(p) <= p_bottom)
    f = f.isel({dim: keep})
    pv = np.asarray(p, dtype=np.float64)[keep]
    axis = f.get_axis_num(dim)

    fv = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    layer = 0.5 * (fv[..., :-1] + fv[..., 1:]) * np.abs(np.diff(pv))
    complete = np.isfinite(layer)
    total = np.where(complete, layer, 0.0).sum(axis=-1) / g
    total[~complete.any(axis=-1)] = np.nan
    return xr.DataArray(total, dims=[d for d in f.dims if d != dim],
                        coords={k: c for k, c in f.coords.items() if dim not in c.dims})


def ddp(f, p, dim="lev"):
    """df/dp (f per Pa) along `dim`, as NCL center_finite_diff_n does it
    (the reference's center_finite_diff_n):

        df/dp[k] = (f[k+1] - f[k-1]) / (p[k+1] - p[k-1])     interior
        df/dp[0] = (f[1] - f[0]) / (p[1] - p[0])             ends, one-sided

    On unevenly spaced levels the interior formula is the slope of the chord
    across two layers. It is exact for straight lines only, unlike the
    three-point formula of the horizontal operators, which is also exact for
    quadratics.

    Args:
        f: field with a level dimension `dim`.
        p: pressure (Pa), the 1D coordinate along `dim` or a field shaped
            like f (on model levels the pressure varies in space).
    """
    axis = f.get_axis_num(dim)
    fv = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    pv = np.moveaxis(np.broadcast_to(_pressure_along(p, f, dim), f.shape), axis, -1)
    out = np.empty_like(fv)
    out[..., 1:-1] = (fv[..., 2:] - fv[..., :-2]) / (pv[..., 2:] - pv[..., :-2])
    out[..., 0] = (fv[..., 1] - fv[..., 0]) / (pv[..., 1] - pv[..., 0])
    out[..., -1] = (fv[..., -1] - fv[..., -2]) / (pv[..., -1] - pv[..., -2])
    return xr.DataArray(np.moveaxis(out, -1, axis), dims=f.dims, coords=f.coords)


class PressureInterpolator:
    """Linear interpolation in pressure, from levels whose pressure varies
    in space (e.g. model levels) to fixed pressure levels.

    For a target pressure p* between two input levels p0 and p1,

        f(p*) = f0 + (f1 - f0) * (p* - p0) / (p1 - p0).

    There is no extrapolation: a target outside the column's pressure range
    (below the lowest model level, or above the highest) is NaN. This is NCL
    linint1_n, as in the reference.

    Finding the bracketing levels is the slow part and depends only on the
    pressure, so it is done once here and reused for every field.

    Args:
        p_in: pressure (Pa) on the input levels, dims (..., `dim`, ...).
        p_out: 1D target pressures (Pa).
        dim: the input level dimension.
        out_dim: name of the output level dimension, with p_out as its
            coordinate; defaults to `dim`.
    """

    def __init__(self, p_in, p_out, dim="lev", out_dim=None):
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.dims = p_in.dims
        self.p_out = np.asarray(p_out, dtype=np.float64)
        axis = p_in.get_axis_num(dim)
        p = np.moveaxis(np.asarray(p_in, dtype=np.float64), axis, -1)
        n = p.shape[-1]
        # Work with pressure increasing along the axis.
        self.flip = bool(np.nanmean(p[..., -1] - p[..., 0]) < 0)
        if self.flip:
            p = p[..., ::-1]

        self.lower, self.weight, self.inside = [], [], []
        for target in self.p_out:
            k0 = np.clip((p <= target).sum(axis=-1) - 1, 0, n - 2)[..., None]
            p0 = np.take_along_axis(p, k0, axis=-1)[..., 0]
            p1 = np.take_along_axis(p, k0 + 1, axis=-1)[..., 0]
            self.lower.append(k0)
            self.weight.append((target - p0) / (p1 - p0))
            self.inside.append((target >= p[..., 0]) & (target <= p[..., -1]))

    def __call__(self, f):
        """f interpolated to p_out, with dims in the same order as f."""
        f = f.transpose(*self.dims)
        axis = f.get_axis_num(self.dim)
        fv = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
        if self.flip:
            fv = fv[..., ::-1]
        out = np.empty(fv.shape[:-1] + (self.p_out.size,))
        for j, (k0, w, inside) in enumerate(zip(self.lower, self.weight, self.inside)):
            f0 = np.take_along_axis(fv, k0, axis=-1)[..., 0]
            f1 = np.take_along_axis(fv, k0 + 1, axis=-1)[..., 0]
            out[..., j] = np.where(inside, f0 + (f1 - f0) * w, np.nan)
        coords = {k: c for k, c in f.coords.items() if self.dim not in c.dims}
        coords[self.out_dim] = (self.out_dim, self.p_out, {"units": "Pa"})
        dims = [self.out_dim if d == self.dim else d for d in f.dims]
        return xr.DataArray(np.moveaxis(out, -1, axis), dims=dims, coords=coords)


def bottom_value(f, p=None, dim="lev"):
    """f at the lowest level that is above ground (the highest pressure
    with a finite value). Columns with no finite value are NaN.

    The reference gets this with NCL linmsg_n (fill missing values, nearest
    value at the ends) and takes the 1000 hPa level; for that level the two
    are the same.
    """
    p = np.asarray(f[dim] if p is None else p, dtype=np.float64)
    order = np.argsort(-p)                       # highest pressure first
    fv = np.moveaxis(np.asarray(f, dtype=np.float64), f.get_axis_num(dim), -1)[..., order]
    finite = np.isfinite(fv)
    first = finite.argmax(axis=-1)[..., None]
    out = np.take_along_axis(fv, first, axis=-1)[..., 0]
    out[~finite.any(axis=-1)] = np.nan
    return xr.DataArray(out, dims=[d for d in f.dims if d != dim],
                        coords={k: c for k, c in f.coords.items() if dim not in c.dims})
