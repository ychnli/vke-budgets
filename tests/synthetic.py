"""Synthetic standard input Datasets with known budgets, for tests."""
import numpy as np
import pandas as pd
import xarray as xr

from vke_budgets.standard import INST_VARS, MEAN_VARS, check_input
from vke_budgets.vertical import GRAVITY

LAT = np.arange(-90.0, 90.1, 7.5)
LON = np.arange(0.0, 360.0, 10.0)
LEV = 100.0 * np.array([1000, 850, 700, 500, 300, 200, 150.0])
PML = 100.0 * np.array([1010, 950, 800, 600, 400, 250, 120, 50.0])   # model levels, bottom-up
U0, Q0, DT = 20.0, 0.01, 10800.0
COLUMN_MASS = 800e2 / GRAVITY                 # 1000-200 hPa, kg m-2


def solid_body(n_time=2, drag=None, drying=None, moistening_inst=1.0, start="2023-01-04"):
    """u = U0 cos(lat), v = omega = 0, q = Q0, phi depending on p only.

    drag: rate r (s-1) of a turbulent drag du/dt = -r u, dv/dt = -r v.
    drying: rate c (kg kg-1 s-1) of a moisture sink dq/dt = -c.
    moistening_inst: q_inst at the end of each window is this factor times
        q_inst at its start (the first step is Q0).
    start: time of the first instantaneous step.
    """
    t_inst = pd.date_range(start, periods=n_time + 1, freq="3h").values
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
