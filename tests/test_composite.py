"""Composites and attribution on fields with known answers."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vke_budgets.composite import ar_index, area_integral, composite, contributions, region

LAT = np.arange(10.0, 70.1, 0.5)
LON = np.arange(-180.0, 180.0, 0.625)
TIMES = pd.date_range("2023-01-01T01:30", periods=200, freq="3h").values


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


def test_ar_index_is_the_box_mean():
    f = field(np.where(np.abs(LAT - 37.75)[:, None] < 2, 5.0, 1.0) * np.ones(LON.size), time=False)
    assert float(ar_index(f, 180.3, 37.75)) == pytest.approx(5.0)


def series(seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.standard_normal(len(TIMES)), dims="time", coords={"time": TIMES})


def test_composite_of_a_linear_field():
    """X = a(x) I(t) + b(x): the composite is a(x) std(I), whatever b."""
    i = series()
    a = np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)
    x = field(a[None] * i.values[:, None, None] + 3.0)
    out = composite(x, i)
    np.testing.assert_allclose(out, a * float(i.std(ddof=1)), rtol=1e-12)


def test_composite_lag_recovers_a_delayed_response():
    """X(t) = I(t - 4): regressing X(t + 4) on I(t) gives back std(I)."""
    i = series(1)
    x = field(np.roll(i.values, 4)[:, None, None] * np.ones((1, LAT.size, LON.size)))
    at_lag = composite(x, i, lag=4)
    expected = float(i.isel(time=slice(0, -4)).std(ddof=1))
    np.testing.assert_allclose(at_lag, expected, rtol=1e-12)
    assert at_lag.attrs["composite_samples"] == len(TIMES) - 4
    assert abs(float(composite(x, i).mean())) < 0.5 * expected     # no response at lag 0


def test_composite_on_selected_times():
    i = series(2)
    when = xr.DataArray(i.values > 0, dims="time", coords={"time": TIMES})
    x = field(2.0 * i.values[:, None, None])
    out = composite(x, i, when=when)
    np.testing.assert_allclose(out, 2.0 * float(i.where(when, drop=True).std(ddof=1)), rtol=1e-12)
    assert out.attrs["composite_samples"] == int(when.sum())


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
