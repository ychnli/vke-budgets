"""Attribution to AR growth/decay and movement, on fields with known answers."""
import numpy as np
import pandas as pd
import xarray as xr

from vke_budgets.contribution import contributions, region

LAT = np.arange(10.0, 70.1, 0.5)
LON = np.arange(-180.0, 180.0, 0.625)
TIMES = pd.date_range("2023-01-01T01:30", periods=3, freq="3h").values


def field(values, time=True):
    dims = ("time", "lat", "lon") if time else ("lat", "lon")
    shape = [len(TIMES), LAT.size, LON.size] if time else [LAT.size, LON.size]
    coords = {"time": TIMES, "lat": LAT, "lon": LON} if time else {"lat": LAT, "lon": LON}
    return xr.DataArray(np.broadcast_to(values, shape).copy(), dims=dims,
                        coords={k: v for k, v in coords.items() if k in dims})


def test_region_crosses_the_dateline():
    f = field(np.zeros((LAT.size, LON.size)), time=False)
    box = region(f, (170, -160), (30, 40))            # 170 E to 160 W
    assert box["lon"].min() == 170 and box["lon"].max() == 200
    assert box.sizes["lat"] == 21


DOMAIN = {"lon": (150.0, 210.0), "lat": (20.0, 50.0)}


def blob(x0, y0=35.0):
    lon360 = LON % 360
    return field(np.exp(-((lon360[None] - x0) ** 2 / 50 + (LAT[:, None] - y0) ** 2 / 20)),
                 time=False)


def test_contributions_separate_growth_and_movement():
    """i is a blob; its tendency is 2 i (growth) plus j, the derivative of
    the blob in x (movement, orthogonal to i). Then the tendency's growth
    contribution is 2 and its movement contribution 1; a term equal to i
    contributes 1 to growth and nothing to movement, one equal to j the
    reverse."""
    i = blob(180.0)
    j = blob(180.5) - blob(179.5)
    terms = xr.Dataset({"tend": 2 * i + j, "growth_only": i, "movement_only": j})
    out = contributions(terms, i, 2 * i + j, **DOMAIN)
    np.testing.assert_allclose(out.growth.sel(term="tend"), 2.0, rtol=1e-6)
    np.testing.assert_allclose(out.movement.sel(term="tend"), 1.0, rtol=1e-6)
    np.testing.assert_allclose(out.growth.sel(term=["growth_only", "movement_only"]), [1, 0],
                               atol=1e-6)
    np.testing.assert_allclose(out.movement.sel(term=["growth_only", "movement_only"]), [0, 1],
                               atol=1e-6)


def test_contributions_add_up():
    """The contributions are linear: those of the parts sum to the whole."""
    rng = np.random.default_rng(3)
    i = blob(185.0)
    parts = {f"p{k}": field(rng.standard_normal((LAT.size, LON.size)), time=False) for k in range(3)}
    whole = sum(parts.values())
    out = contributions(xr.Dataset(parts), i, whole, **DOMAIN)
    total = contributions(xr.Dataset({"whole": whole}), i, whole, **DOMAIN)
    np.testing.assert_allclose(out.growth.sum("term"), total.growth.sel(term="whole"), rtol=1e-10)
    np.testing.assert_allclose(out.movement.sum("term"), 1.0, rtol=1e-10)


def test_contributions_frame_by_frame():
    """With a time dimension, each frame is attributed on its own."""
    rng = np.random.default_rng(4)
    i = xr.concat([blob(x0) for x0 in (175.0, 180.0, 185.0)], "time").assign_coords(time=TIMES)
    tend = field(rng.standard_normal((len(TIMES), LAT.size, LON.size)))
    terms = xr.Dataset({"half": tend / 2, "rest": tend / 2})
    out = contributions(terms, i, tend, **DOMAIN)
    assert out.growth.sizes["time"] == 3
    for k in range(3):
        one = contributions(terms.isel(time=k), i.isel(time=k), tend.isel(time=k), **DOMAIN)
        xr.testing.assert_allclose(out.isel(time=k), one)
