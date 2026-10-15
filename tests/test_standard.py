"""check_input on a small synthetic standard Dataset."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vke_budgets.standard import INST_VARS, MEAN_VARS, PHYSICS_VARS, check_input, physics_processes


def synthetic(n_time=2, physics=True):
    """A tiny standard input Dataset filled with zeros."""
    sizes = {"time": n_time, "time_inst": n_time + 1, "lev": 3, "mlev": 4, "lat": 5, "lon": 8}
    t_inst = pd.date_range("2023-01-04", periods=n_time + 1, freq="3h").values
    coords = {
        "time_inst": t_inst,
        "time": t_inst[:-1] + np.timedelta64(90, "m"),
        "lev": ("lev", [1000e2, 850e2, 500e2], {"units": "Pa"}),
        "mlev": np.arange(4, 0, -1),
        "lat": np.linspace(-60, 60, 5),
        "lon": np.arange(0, 360, 45.0),
    }
    specs = {**INST_VARS, **MEAN_VARS, **(PHYSICS_VARS if physics else {})}
    data = {name: (dims, np.zeros([sizes[d] for d in dims], dtype=np.float32), {"units": units})
            for name, (dims, units) in specs.items()}
    return xr.Dataset(data, coords=coords)


def test_synthetic_passes():
    check_input(synthetic())
    check_input(synthetic(physics=False))
    assert physics_processes(synthetic(physics=False)) == ((), ())


def test_problems_are_all_reported():
    ds = synthetic().drop_vars("omega")
    ds["q"].attrs["units"] = "g/kg"
    with pytest.raises(ValueError) as err:
        check_input(ds)
    assert "missing omega" in str(err.value) and "q has units 'g/kg'" in str(err.value)


def test_times_must_bracket_windows():
    ds = synthetic()
    ds = ds.assign_coords(time=ds["time"] + np.timedelta64(3, "h"))
    with pytest.raises(ValueError, match="between consecutive time_inst"):
        check_input(ds)


def test_partial_momentum_process_does_not_count():
    ds = synthetic().drop_vars("dvdt_gwd")
    assert "gwd" not in physics_processes(ds)[0]
