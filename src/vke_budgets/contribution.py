"""
Compute how much each budget term contributes to the growth, decay and movement 
of an atmospheric river. 

Over a domain S around the AR, denoting I the column KEV or VTE and dI/dt its 
tendency in one frame (one time step), a term X contributes:

    growth/decay:  G(X) = int_S X I dS / int_S I^2 dS                 (Eq 8)
    movement:      M(X) = int_S X I~ dS / int_S I~^2 dS               (Eq 9)
                   I~ = dI/dt - G(dI/dt) I                            (Eq 10)

G(X) is the projection of X onto the AR pattern I 
I~ is the tendency with its uniform growth or decay removed 
M(X) is the share of the movement tendency that X explains.

See eq 8-10 in Zhang et al (2026) for more details.
"""
import numpy as np
import xarray as xr


def region(da: xr.DataArray, lon: tuple[float, float],
           lat: tuple[float, float]) -> xr.DataArray:
    """da within lon = (west, east) and lat = (south, north), in degrees.
    Longitudes are read as degrees east, 0-360; west > east crosses 0 E."""
    da = da.assign_coords(lon=da["lon"] % 360).sortby("lon").sortby("lat")
    west, east = (x % 360 for x in lon)
    inside = (da["lon"] >= west) & (da["lon"] <= east) if west <= east else \
        (da["lon"] >= west) | (da["lon"] <= east)
    return da.sel(lon=inside, lat=slice(*sorted(lat)))


def area_integral(da: xr.DataArray) -> xr.DataArray:
    """Integral over the lat-lon points of da, weighted by cos(latitude)
    (proportional to dS; the constant factors cancel in every ratio here)."""
    return (da * np.cos(np.deg2rad(da["lat"]))).sum(("lat", "lon"))


def contributions(terms: xr.Dataset, i: xr.DataArray, i_tend: xr.DataArray,
                  lon: tuple[float, float], lat: tuple[float, float]) -> xr.Dataset:
    """Contribution of each term to the growth/decay and to the movement of
    the AR variable i over the domain (lon, lat), Eqs. (8)-(10).

    Each frame is treated on its own: the integrals run over lat and lon
    only, so inputs with a `time` dimension give one value per time step.

    Args:
        terms: Dataset of column-integrated tendency terms, e.g. the result
            of `budget.groups`.
        i: the AR variable, e.g. the column KEV.
        i_tend: its tendency, e.g. the `tend` group.
        lon, lat: (west, east) and (south, north) of the domain S.
    Returns:
        Dataset with `growth` (s-1; times 86400 for day-1) and `movement`
        (1) on a `term` dimension.
    """
    def project(x: xr.DataArray, onto: xr.DataArray) -> xr.DataArray:
        return area_integral(region(x, lon, lat) * region(onto, lon, lat)) \
            / area_integral(region(onto, lon, lat) ** 2)

    i_tilde = i_tend - project(i_tend, i) * i            # movement tendency, Eq. (10)
    names = list(terms.data_vars)
    growth = [project(terms[n], i) for n in names]
    movement = [project(terms[n], i_tilde) for n in names]
    return xr.Dataset({
        "growth": xr.concat(growth, "term").assign_attrs(
            units="s-1", long_name="contribution to AR growth/decay"),
        "movement": xr.concat(movement, "term").assign_attrs(
            units="1", long_name="contribution to AR movement"),
    }).assign_coords(term=names)
