"""
Budget assembly on synthetic standard input with known answers: solid-body
rotation with uniform humidity, where every dynamical term vanishes, plus
simple physics tendencies whose terms are known in closed form.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vke_budgets import budget
from vke_budgets.standard import INST_VARS, MEAN_VARS, check_input
from vke_budgets.vertical import GRAVITY

LAT = np.arange(-90.0, 90.1, 7.5)
LON = np.arange(0.0, 360.0, 10.0)
LEV = 100.0 * np.array([1000, 850, 700, 500, 300, 200, 150.0])
PML = 100.0 * np.array([1010, 950, 800, 600, 400, 250, 120, 50.0])   # model levels, bottom-up
U0, Q0, DT = 20.0, 0.01, 10800.0
COLUMN_MASS = 800e2 / GRAVITY                 # 1000-200 hPa, kg m-2


def solid_body(n_time=2, drag=None, drying=None, moistening_inst=1.0):
    """u = U0 cos(lat), v = omega = 0, q = Q0, phi depending on p only.

    drag: rate r (s-1) of a turbulent drag du/dt = -r u, dv/dt = -r v.
    drying: rate c (kg kg-1 s-1) of a moisture sink dq/dt = -c.
    moistening_inst: q_inst at the end of each window is this factor times
        q_inst at its start (the first step is Q0).
    """
    t_inst = pd.date_range("2023-01-04", periods=n_time + 1, freq="3h").values
    coslat = np.cos(np.deg2rad(LAT))[:, None]

    def field(dims, value):
        shape = {"time": n_time, "time_inst": n_time + 1, "lev": LEV.size,
                 "mlev": PML.size, "lat": LAT.size, "lon": LON.size}
        return np.broadcast_to(value, [shape[d] for d in dims]).astype(np.float32).copy()

    data = {}
    for name, (dims, units) in {**INST_VARS, **MEAN_VARS}.items():
        value = {"q_inst": Q0, "u_inst": U0 * coslat, "v_inst": 0.0, "ps_inst": 1013e2,
                 "p": PML[:, None, None], "q": Q0, "u": U0 * coslat, "v": 0.0,
                 "phi": 9.81 * 7000.0 * np.log(1000e2 / PML)[:, None, None],
                 "omega": 0.0}[name]
        data[name] = (dims, field(dims, value), {"units": units})
    growth = moistening_inst ** np.arange(n_time + 1)
    data["q_inst"][1][...] *= growth[:, None, None, None]
    if drag is not None:
        for c in "uv":
            mean = data[c][1]
            data[f"d{c}dt_turb"] = (("time", "lev", "lat", "lon"),
                                    field(("time", "lev", "lat", "lon"), -drag * mean[:, :1]),
                                    {"units": "m s-2"})
    if drying is not None:
        data["dqdt_moist"] = (("time", "lev", "lat", "lon"),
                              field(("time", "lev", "lat", "lon"), -drying),
                              {"units": "kg kg-1 s-1"})
    coords = {"time_inst": t_inst, "time": t_inst[:-1] + np.timedelta64(90, "m"),
              "lev": ("lev", LEV, {"units": "Pa"}), "mlev": np.arange(PML.size),
              "lat": LAT, "lon": LON}
    ds = xr.Dataset(data, coords=coords)
    check_input(ds)
    return ds


def kinetic_energy():
    return 0.5 * (U0 * np.cos(np.deg2rad(LAT))[:, None]) ** 2 * np.ones((1, LON.size))


INNER = np.abs(LAT) < 87


def assert_close(actual, expected, **kw):
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), **kw)


def test_dynamical_terms_vanish_for_solid_body_rotation():
    out = budget.compute(solid_body())
    scale = 2 * Q0 * kinetic_energy().max() * COLUMN_MASS / DT   # a generous term size
    for quantity in ("", "_v"):
        for term in ("IHAKE", "IVAKE", "IPEKE", "IHAV", "IVAV", "IHCV", "IVCV"):
            values = out[term + quantity].values[:, INNER]
            np.testing.assert_allclose(values, 0.0, atol=1e-12 * scale, err_msg=term + quantity)


def test_column_quantities():
    out = budget.compute(solid_body())
    k = kinetic_energy()
    assert_close(out["IKEVAVG"].values[:, INNER], (Q0 * k * COLUMN_MASS)[INNER], rtol=1e-6)
    assert_close(out["IVTEAVG"].values[:, INNER], (Q0**2 * k * COLUMN_MASS)[INNER], rtol=1e-6)
    assert_close(out["IVT"].values, Q0 * U0 * np.cos(np.deg2rad(LAT))[:, None] * COLUMN_MASS,
                 rtol=1e-6)
    # Window outputs are blanked at the poles; instantaneous ones are not.
    assert np.isnan(out["IKEVAVG"].values[:, ~INNER]).all()
    assert np.isfinite(out["IKEV"].values).all()


def test_physics_terms_have_their_closed_form():
    """Drag du/dt = -r u gives TOKE = q u.(-r u) = -2 r q K, and for VTE
    -2 r q^2 K. A sink dq/dt = -c gives POV = -c K, and for VTE -2 c q K."""
    r, c = 1e-5, 1e-8
    out = budget.compute(solid_body(drag=r, drying=c))
    k = kinetic_energy()[INNER]
    expected = {"ITOKE": -2 * r * Q0 * k, "ITOKE_v": -2 * r * Q0**2 * k,
                "IPOV": -c * k, "IPOV_v": -2 * c * Q0 * k}
    for name, value in expected.items():
        assert_close(out[name].values[:, INNER], value * COLUMN_MASS, rtol=1e-6, err_msg=name)
    # Only the physics present in the input is computed.
    assert "IMOKE" not in out and "IAOKEV" not in out


def test_tendency_of_moistening_column():
    """q_inst grows by a factor f over each window: IKEVTEND = (f - 1) <q K>/dt
    for the start of the window, and the local tendency integrates to the same
    thing with the window-mean q."""
    f = 1.1
    out = budget.compute(solid_body(moistening_inst=f))
    k = kinetic_energy()[INNER]
    start = Q0 * f ** np.arange(2)[:, None, None]
    np.testing.assert_allclose(out["IKEVTEND"].values[:, INNER],
                               (f - 1) * start * k * COLUMN_MASS / DT, rtol=1e-5)
    np.testing.assert_allclose(out["I_KEVTEND"].values[:, INNER],
                               k * (f - 1) * start * COLUMN_MASS / DT, rtol=1e-5)
    # The surface pressure does not change.
    np.testing.assert_array_equal(out["SPTE"].values[:, INNER], 0.0)


def test_window_selection_and_options():
    ds = solid_body(n_time=3)
    out = budget.compute(ds, windows=[1])
    assert out.sizes["time"] == 1 and out.sizes["time_inst"] == 2
    np.testing.assert_array_equal(out["time"], ds["time"][1:2])
    np.testing.assert_array_equal(out["time_inst"], ds["time_inst"][1:3])

    out = budget.compute(ds, budget.BudgetOptions(quantities=("KEV",), keep_3d=True), windows=[0])
    assert "IHAKE_v" not in out and "IVTE" not in out
    assert out["HAKE"].dims == ("time", "lev", "lat", "lon")
    assert out["HAKE"].attrs["units"] == "m2 s-3" and out["IHAKE"].attrs["units"] == "kg s-3"


def test_threads_give_the_same_answer():
    ds = solid_body(n_time=3, drag=1e-5)
    xr.testing.assert_identical(budget.compute(ds), budget.compute(ds, n_workers=3))


def test_groups_and_residual():
    """With drag only, fric = ITOKE, every other group is 0, tend = 0, so
    resid = -ITOKE. Groups missing from the input count as zero."""
    out = budget.compute(solid_body(drag=1e-5))
    for quantity, sfx in (("KEV", ""), ("VTE", "_v")):
        g = budget.groups(out, quantity)
        assert list(g.data_vars) == ["tend", "hflux", "vflux", "pe", "cond", "fric", "ana",
                                     "spte", "resid"]
        xr.testing.assert_allclose(g.fric, out["ITOKE" + sfx])
        xr.testing.assert_allclose(g.resid, g.tend - sum(g[k] for k in list(g.data_vars)[1:-1]))
        np.testing.assert_allclose(g.resid.values[:, INNER], -out["ITOKE" + sfx].values[:, INNER],
                                   rtol=1e-6, atol=1e-12)
        assert float(np.abs(g.cond).max()) == 0.0
