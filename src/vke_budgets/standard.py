"""
The standard input Dataset: this is what the budget code expects, from any data
source. A source adapter (e.g. sources/merra2.py) turns raw files into this
format so that the physics doesn't have to care about source-specific names
units or pressure level orders.

Coordinates
    time       centres of the averaging windows (datetime64)
    time_inst  instantaneous times, one more than `time`. Window i runs from
               time_inst[i] to time_inst[i+1].
    lev        pressure levels (Pa)
    mlev       model levels, ordered bottom to top (the label is the
               source's own level number)
    lat, lon   degrees north and east, 1D. Latitude may run either way.

Instantaneous state on pressure levels (dims time_inst, lev, lat, lon;
NaN below the ground)
    q_inst     specific humidity (kg kg-1)
    u_inst     eastward wind (m s-1)
    v_inst     northward wind (m s-1)
    ps_inst    surface pressure (Pa), dims (time_inst, lat, lon)

Window-mean state on model levels (dims time, mlev, lat, lon)
    p          pressure at the model levels (Pa)
    q, u, v    as above
    phi        geopotential (m2 s-2)
    omega      vertical pressure velocity dp/dt (Pa s-1)

Window-mean physics tendencies on pressure levels (dims time, lev, lat, lon;
optional, as not every source has them). Units m s-2 for u and v, kg kg-1 s-1
for q.
    dudt_<process>, dvdt_<process>  for process in MOMENTUM_PROCESSES
    dqdt_<process>                  for process in MOISTURE_PROCESSES

Processes: moist (convection, large-scale condensation and precipitation),
turb (turbulence), gwd (gravity-wave drag), chem (chemistry), ana (the analysis increment), dyn (the model's
own dynamics, a cross-check on the dynamical terms computed offline).
"""
import numpy as np

MOMENTUM_PROCESSES = ("moist", "turb", "gwd", "dyn", "ana")
MOISTURE_PROCESSES = ("moist", "turb", "chem", "dyn", "ana")

INST_VARS = {
    "q_inst": (("time_inst", "lev", "lat", "lon"), "kg kg-1"),
    "u_inst": (("time_inst", "lev", "lat", "lon"), "m s-1"),
    "v_inst": (("time_inst", "lev", "lat", "lon"), "m s-1"),
    "ps_inst": (("time_inst", "lat", "lon"), "Pa"),
}
MEAN_VARS = {
    name: (("time", "mlev", "lat", "lon"), units)
    for name, units in [("p", "Pa"), ("q", "kg kg-1"), ("u", "m s-1"), ("v", "m s-1"),
                        ("phi", "m2 s-2"), ("omega", "Pa s-1")]
}
PHYSICS_VARS = {
    **{f"{c}dt_{proc}": (("time", "lev", "lat", "lon"), "m s-2")
       for proc in MOMENTUM_PROCESSES for c in ("du", "dv")},
    **{f"dqdt_{proc}": (("time", "lev", "lat", "lon"), "kg kg-1 s-1")
       for proc in MOISTURE_PROCESSES},
}


def check_input(ds):
    """Raise ValueError, listing every problem, if ds is not a standard
    input Dataset. Physics tendencies are optional, but if present they must
    have the right dims and units."""
    problems = []
    for name, (dims, units) in {**INST_VARS, **MEAN_VARS, **PHYSICS_VARS}.items():
        if name not in ds:
            if name not in PHYSICS_VARS:
                problems.append(f"missing {name}")
            continue
        if set(ds[name].dims) != set(dims):
            problems.append(f"{name} has dims {ds[name].dims}, expected {dims}")
        if ds[name].attrs.get("units") != units:
            problems.append(f"{name} has units {ds[name].attrs.get('units')!r}, "
                            f"expected {units!r}")
    for coord in ("time", "time_inst", "lev", "mlev", "lat", "lon"):
        if coord not in ds.coords:
            problems.append(f"missing coordinate {coord}")
    if not problems:
        if ds.sizes["time_inst"] != ds.sizes["time"] + 1:
            problems.append("time_inst must have one more step than time")
        else:
            t0, t1 = ds["time_inst"].values[:-1], ds["time_inst"].values[1:]
            if not np.all((t0 < ds["time"].values) & (ds["time"].values < t1)):
                problems.append("each time must lie between consecutive time_inst")
        if ds["lev"].attrs.get("units") != "Pa":
            problems.append("lev must be in Pa")
    if problems:
        raise ValueError("not a standard input Dataset:\n  " + "\n  ".join(problems))


def physics_processes(ds):
    """The physics tendencies present in ds, as (momentum, moisture) tuples
    of process names. A momentum process counts only if both components are
    there."""
    momentum = tuple(p for p in MOMENTUM_PROCESSES
                     if f"dudt_{p}" in ds and f"dvdt_{p}" in ds)
    moisture = tuple(p for p in MOISTURE_PROCESSES if f"dqdt_{p}" in ds)
    return momentum, moisture
