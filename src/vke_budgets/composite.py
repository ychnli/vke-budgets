"""
Regression composites of atmospheric rivers, and how each budget term
contributes to their growth, decay and movement.

Composite (Ong & Yang 2024, Methods; Zhang et al. 2026, Section 2.3). Pick
a point in an AR region and take as the AR index I0(t) the mean of the
column KEV in a 1 x 1 degree box around it. Then, at every grid point,
regress each field X(t) on the index and scale by one standard deviation of
the index:

    composite(X) = beta * std(I0),   beta = sum_t X'(t) I0'(t) / sum_t I0'(t)^2

where ' marks the departure from the time mean. The composite is the
anomaly of X that comes with a one-standard-deviation AR at the point. With
a lag L, X is taken L steps after the index (X(t + L) regressed on I0(t)),
which gives the AR's evolution before and after. Zhang et al. use only the
times with an AR at the point (`when`).

Attribution (Zhang et al. 2026, Eqs. 8-10). Over the composite domain S,
with I the composite column KEV and dI/dt its composite tendency, a term X
contributes to

    growth/decay:  G(X) = int_S X I dS / int_S I^2 dS                 (8)
    movement:      M(X) = int_S X I~ dS / int_S I~^2 dS               (9)
                   I~ = dI/dt - G(dI/dt) I                            (10)

I~ is the tendency with its uniform growth or decay removed (a Gram-Schmidt
step, so I~ is orthogonal to I): what is left moves the AR. G is a rate
(s-1); M is the share of the movement tendency. Both are linear in X, so
the contributions of the terms add up to those of the total tendency.
"""
import numpy as np
import xarray as xr


def region(da, lon, lat):
    """da within lon = (west, east) and lat = (south, north), in degrees.
    Longitudes are read as degrees east, 0-360; west > east crosses 0 E."""
    da = da.assign_coords(lon=da["lon"] % 360).sortby("lon").sortby("lat")
    west, east = (x % 360 for x in lon)
    inside = (da["lon"] >= west) & (da["lon"] <= east) if west <= east else \
        (da["lon"] >= west) | (da["lon"] <= east)
    return da.sel(lon=inside, lat=slice(*sorted(lat)))


def area_integral(da):
    """Integral over the lat-lon points of da, weighted by cos(latitude)
    (proportional to dS; the constant factors cancel in every ratio here)."""
    return (da * np.cos(np.deg2rad(da["lat"]))).sum(("lat", "lon"))


def ar_index(field, lon, lat, size=1.0):
    """Area mean of `field` in a size x size degree box centred at (lon, lat):
    the AR index, a time series."""
    half = size / 2
    box = region(field, (lon - half, lon + half), (lat - half, lat + half))
    return area_integral(box) / area_integral(xr.ones_like(box))


def composite(fields, index, lag=0, when=None):
    """Regression composite of `fields` on `index`: the regression
    coefficient times one standard deviation of the index (see above).

    Args:
        fields: DataArray or Dataset on `time` (e.g. budget terms).
        index: the AR index on the same, evenly spaced `time`.
        lag: number of time steps the fields lag the index (negative: lead).
        when: optional boolean series on `time`: use only these index times
            (e.g. an AR at the composite point).
    """
    step = np.diff(index["time"].values)
    if step.size and not np.all(step == step[0]):
        raise ValueError("composite needs an evenly spaced time series")
    n = index.sizes["time"]
    # Pair the index at t with the fields at t + lag.
    i_times = slice(max(0, -lag), n - max(0, lag))
    f_times = slice(max(0, lag), n - max(0, -lag))
    index = index.isel(time=i_times)
    fields = fields.isel(time=f_times).assign_coords(time=index["time"])
    if when is not None:
        keep = when.sel(time=index["time"]).values
        index, fields = index.isel(time=keep), fields.isel(time=keep)

    index_anomaly = index - index.mean("time")
    # min_count: a point missing at every time (e.g. a blanked pole) stays NaN.
    beta = ((fields - fields.mean("time")) * index_anomaly).sum("time", min_count=1) \
        / (index_anomaly**2).sum("time")
    out = beta * index.std("time", ddof=1)
    return out.assign_attrs(composite_lag=lag, composite_samples=index.sizes["time"])


def contributions(terms, i, i_tend, lon, lat):
    """Contribution of each term to the growth/decay and to the movement of
    the AR variable i over the domain (lon, lat), Eqs. (8)-(10).

    Args:
        terms: Dataset of (composite) column-integrated tendency terms.
        i: the (composite) AR variable, e.g. the column KEV.
        i_tend: its (composite) tendency.
        lon, lat: (west, east) and (south, north) of the domain S.
    Returns:
        Dataset with `growth` (s-1; times 86400 for day-1) and `movement`
        (1) on a `term` dimension.
    """
    def project(x, onto):
        return area_integral(region(x, lon, lat) * region(onto, lon, lat)) \
            / area_integral(region(onto, lon, lat) ** 2)

    i_tilde = i_tend - project(i_tend, i) * i            # movement tendency, Eq. (10)
    names = list(terms.data_vars)
    growth = [project(terms[n], i) for n in names]
    movement = [project(terms[n], i_tilde) for n in names]
    return xr.Dataset({
        "growth": xr.concat(growth, "term").assign_attrs(
            units="s-1", long_name="contribution to AR growth/decay, Eq. (8)"),
        "movement": xr.concat(movement, "term").assign_attrs(
            units="1", long_name="contribution to AR movement, Eq. (9)"),
    }).assign_coords(term=names)
