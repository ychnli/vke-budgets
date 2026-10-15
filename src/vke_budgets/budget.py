"""
Assemble the KEV and VTE budgets from a standard input Dataset
(standard.py), one averaging window at a time.

For each window:

1. On model levels, form the vertical-derivative pieces omega du/dp,
   omega dv/dp, omega dq/dp and d(q^n omega)/dp. Model levels never go below
   the ground, and their spacing is finest near the surface.
2. Interpolate the window-mean state and those pieces to pressure levels.
3. On pressure levels, form every term of terms.py, taking horizontal
   derivatives along pressure surfaces.
4. Integrate each term over the column, p_bottom to p_top (1000-200 hPa).
5. Add the tendencies from the instantaneous fields at the two ends of the
   window: I{E}TEND = (<E>(end) - <E>(start)) / dt, and SPTE.

The budget closes as

    I{E}TEND = IHAKE + IHCV + IVAKE + IVCV + IPEKE + (physics) + IAOKEV
               + SPTE + residual,

where (physics) is the sum of the momentum and moisture physics terms
present in the input. Output names follow the reference: an "I" prefix
marks a column integral, and the VTE budget carries the suffix "_v".
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import xarray as xr

from . import terms
from .geometry import DEFAULT, HorizontalGeometry
from .standard import check_input, physics_processes
from .vertical import GRAVITY, PressureInterpolator, bottom_value, ddp, integrate_column

# Budgeted quantities E = q^n K: name -> n.
QUANTITIES = {"KEV": 1, "VTE": 2}

# Physics tendencies -> term names (before the VTE suffix).
MOMENTUM_TERMS = {"moist": "MOKE", "turb": "TOKE", "gwd": "GOKE", "dyn": "DOKE"}
MOISTURE_TERMS = {"moist": "POV", "turb": "TOV", "chem": "COV", "dyn": "DOV"}

LONG_NAMES = {
    "HAKE": "horizontal advection of KE",
    "VAKE": "vertical advection of KE",
    "PEKE": "potential energy conversion to KE",
    "HAV": "horizontal advection of vapor",
    "VAV": "vertical advection of vapor",
    "HCV": "horizontal convergence of vapor transport, -K div(q^n u)",
    "VCV": "vertical convergence of vapor transport, -K d(q^n omega)/dp",
    "HCV_vO": "horizontal convergence, -2qK div(q u) (NCL commented-out form)",
    "VCV_vO": "vertical convergence, -2qK d(q omega)/dp (NCL commented-out form)",
    "MOKE": "moist processes on KE",
    "TOKE": "turbulence on KE",
    "GOKE": "gravity-wave drag on KE",
    "DOKE": "MERRA-2 dynamics on KE",
    "POV": "moist processes on vapor",
    "TOV": "turbulence on vapor",
    "COV": "chemistry on vapor",
    "DOV": "MERRA-2 dynamics on vapor",
    "AOKEV": "analysis increments",
}


@dataclass(frozen=True)
class BudgetOptions:
    """What to compute, and how.

    Attributes:
        geometry: horizontal operators; geometry.REFERENCE reproduces the
            reference.
        quantities: which budgets, from QUANTITIES.
        g: gravity (m s-2) in the column integrals.
        p_bottom, p_top: column integral limits (Pa).
        pole_lat_limit: window outputs poleward of this latitude (degrees)
            are set to NaN, where the lat-lon grid is too distorted for
            finite differences. None keeps everything.
        keep_3d: also return the terms on pressure levels.
        dyn_crosscheck: also compute DOKE and DOV from the model's own
            dynamics tendencies, a cross-check on HAKE + VAKE + PEKE etc.
        ncl_vte_form: also compute HCV_vO and VCV_vO for VTE.
    """
    geometry: HorizontalGeometry = DEFAULT
    quantities: tuple = ("KEV", "VTE")
    g: float = GRAVITY
    p_bottom: float = 1000e2
    p_top: float = 200e2
    pole_lat_limit: float | None = 87.0
    keep_3d: bool = False
    dyn_crosscheck: bool = False
    ncl_vte_form: bool = False


def _name(term, quantity):
    """Output name of `term` in the budget of `quantity`."""
    return term if quantity == "KEV" else f"{term}_v"


def instantaneous_columns(inst, options=BudgetOptions()):
    """Column integrals of the instantaneous state (dims time_inst, lat, lon):
    IVT = |<q u>| (kg m-1 s-1), and I<E> = <q^n K> (kg s-2) for each budget."""
    integ = _integrator(options)
    q, u, v = inst["q_inst"], inst["u_inst"], inst["v_inst"]
    k = terms.kinetic_energy(u, v)
    out = {"IVT": np.hypot(integ(q * u), integ(q * v)).assign_attrs(
        units="kg m-1 s-1", long_name="integrated vapor transport")}
    for quantity in options.quantities:
        out[f"I{quantity}"] = integ(q ** QUANTITIES[quantity] * k).assign_attrs(
            units="kg s-2", long_name=f"column {quantity}, q^{QUANTITIES[quantity]} K")
    return xr.Dataset(out)


def _integrator(options):
    def integ(f):
        return integrate_column(f, dim="lev", g=options.g,
                                p_bottom=options.p_bottom, p_top=options.p_top)
    return integ


def window_budget(window, inst_start, inst_end, dt, options=BudgetOptions()):
    """Every budget term for one averaging window.

    Args:
        window: the standard input at one `time` (the window-mean state and
            physics tendencies), loaded.
        inst_start, inst_end: the instantaneous state at the two ends of the
            window, with their column integrals (instantaneous_columns)
            merged in.
        dt: window length (s).
    Returns:
        Dataset of column integrals (lat, lon), plus the 3D terms on pressure
        levels if options.keep_3d.
    """
    geom, integ = options.geometry, _integrator(options)

    # 1. Vertical-derivative pieces on model levels.
    p = window["p"]
    q_ml, u_ml, v_ml, omega_ml = window["q"], window["u"], window["v"], window["omega"]
    pieces = {
        "omega_dudp": omega_ml * ddp(u_ml, p, "mlev"),
        "omega_dvdp": omega_ml * ddp(v_ml, p, "mlev"),
        "omega_dqdp": omega_ml * ddp(q_ml, p, "mlev"),
    }
    for quantity in options.quantities:
        n = QUANTITIES[quantity]
        pieces[f"d_q{n}omega_dp"] = ddp(q_ml**n * omega_ml, p, "mlev")
    if options.ncl_vte_form and "d_q1omega_dp" not in pieces:
        pieces["d_q1omega_dp"] = ddp(q_ml * omega_ml, p, "mlev")

    # 2. To pressure levels.
    to_p = PressureInterpolator(p, window["lev"].values, dim="mlev", out_dim="lev")
    q, u, v, phi = (to_p(window[name]) for name in ("q", "u", "v", "phi"))
    pieces = {name: to_p(f) for name, f in pieces.items()}
    k = terms.kinetic_energy(u, v)

    # Changes of the instantaneous state across the window.
    dqdt = (inst_end["q_inst"] - inst_start["q_inst"]) / dt
    dudt = (inst_end["u_inst"] - inst_start["u_inst"]) / dt
    dvdt = (inst_end["v_inst"] - inst_start["v_inst"]) / dt
    dpsdt = (inst_end["ps_inst"] - inst_start["ps_inst"]) / dt

    momentum, moisture = physics_processes(window)
    out = {}
    for quantity in options.quantities:
        n = QUANTITIES[quantity]
        w_m, w_v = terms.weights(q, k, n)
        e = w_m * k

        # 3. Terms on pressure levels (m2 s-3).
        fields = {
            "HAKE": terms.horizontal_advection_of_ke(w_m, u, v, k, geom),
            "VAKE": terms.vertical_advection_of_ke(w_m, u, v, pieces["omega_dudp"],
                                                   pieces["omega_dvdp"]),
            "PEKE": terms.pe_to_ke_conversion(w_m, u, v, phi, geom),
            "HAV": terms.horizontal_advection_of_vapor(w_v, u, v, q, geom),
            "VAV": terms.vertical_advection_of_vapor(w_v, pieces["omega_dqdp"]),
            "HCV": terms.horizontal_convergence_of_vapor(k, w_m, u, v, geom),
            "VCV": terms.vertical_convergence_of_vapor(k, pieces[f"d_q{n}omega_dp"]),
        }
        for proc, term in MOMENTUM_TERMS.items():
            if proc in momentum and (proc != "dyn" or options.dyn_crosscheck):
                fields[term] = terms.momentum_physics(w_m, u, v, window[f"dudt_{proc}"],
                                                      window[f"dvdt_{proc}"])
        for proc, term in MOISTURE_TERMS.items():
            if proc in moisture and (proc != "dyn" or options.dyn_crosscheck):
                fields[term] = terms.moisture_physics(w_v, window[f"dqdt_{proc}"])
        if "ana" in momentum and "ana" in moisture:
            fields["AOKEV"] = (terms.momentum_physics(w_m, u, v, window["dudt_ana"],
                                                      window["dvdt_ana"])
                               + terms.moisture_physics(w_v, window["dqdt_ana"]))
        if options.ncl_vte_form and n == 2:
            fields["HCV_vO"], fields["VCV_vO"] = terms.ncl_convergence_of_vapor(
                w_v, q, u, v, pieces["d_q1omega_dp"], geom)
        tend = terms.local_tendency(w_m, w_v, u, v, dqdt, dudt, dvdt)

        # 4. Column integrals (kg s-3).
        for term, f in fields.items():
            name = term if term.endswith("_vO") else _name(term, quantity)
            out[f"I{name}"] = integ(f).assign_attrs(
                units="kg s-3", long_name=f"I{quantity} tendency, {LONG_NAMES[term]}")
            if options.keep_3d:
                out[name] = f.assign_attrs(
                    units="m2 s-3", long_name=f"{quantity} tendency, {LONG_NAMES[term]}")
        out[f"I{quantity}AVG"] = integ(e).assign_attrs(
            units="kg s-2", long_name=f"column {quantity}, window mean")
        out[f"I_{quantity}TEND"] = integ(tend).assign_attrs(
            units="kg s-3", long_name=f"integral of the local {quantity} tendency")

        # 5. Tendency of the column integral, and the boundary term.
        out[f"I{quantity}TEND"] = ((inst_end[f"I{quantity}"] - inst_start[f"I{quantity}"]) / dt
                                   ).assign_attrs(units="kg s-3",
                                                  long_name=f"I{quantity} tendency")
        out[_name("SPTE", quantity)] = terms.surface_pressure_term(
            dpsdt, bottom_value(e, dim="lev"), options.g).assign_attrs(
            units="kg s-3", long_name=f"surface pressure tendency effect on I{quantity}")
        if options.keep_3d:
            out[quantity] = e.assign_attrs(units="m2 s-2", long_name=f"{quantity}, q^{n} K")
            out[f"{quantity}TEND"] = tend.assign_attrs(
                units="m2 s-3", long_name=f"local {quantity} tendency")

    result = xr.Dataset(out)
    if options.pole_lat_limit is not None:
        result = result.where(np.abs(result["lat"]) < options.pole_lat_limit)
    return result


INST_INPUTS = ["q_inst", "u_inst", "v_inst", "ps_inst"]


def compute(ds, options=BudgetOptions(), windows=None, n_workers=1, progress=False):
    """The budgets for the windows of a standard input Dataset.

    Args:
        ds: standard input Dataset (e.g. sources.merra2.open_day), lazy or
            in memory.
        options: BudgetOptions.
        windows: indices of the windows to compute (default: all).
        n_workers: windows computed at once, in threads. Each window needs a
            few GB.
        progress: print one line per window.
    Returns:
        Dataset with the window terms on `time` and the instantaneous column
        integrals (IVT, IKEV, IVTE) on `time_inst`, for the selected windows
        and the instantaneous steps that bound them.
    """
    check_input(ds)
    windows = range(ds.sizes["time"]) if windows is None else list(windows)
    steps = sorted({j for i in windows for j in (i, i + 1)})
    inst = ds[INST_INPUTS].isel(time_inst=steps).load()
    inst = inst.merge(instantaneous_columns(inst, options))
    seconds = (ds["time_inst"].values[1:] - ds["time_inst"].values[:-1]) / np.timedelta64(1, "s")

    def one(i):
        window = ds.drop_dims("time_inst").isel(time=i).load()
        start = inst.sel(time_inst=ds["time_inst"].values[i])
        end = inst.sel(time_inst=ds["time_inst"].values[i + 1])
        result = window_budget(window, start, end, seconds[i], options)
        if progress:
            print(f"  window {i}: {str(window['time'].values)[:16]}", flush=True)
        result = result.drop_vars(["time", "time_inst"], errors="ignore")
        return result.expand_dims(time=[window["time"].values])

    if n_workers > 1:
        with ThreadPoolExecutor(n_workers) as pool:
            results = list(pool.map(one, windows))
    else:
        results = [one(i) for i in windows]

    columns = inst[["IVT"] + [f"I{qn}" for qn in options.quantities]]
    out = xr.merge([xr.concat(results, dim="time"), columns])
    out.attrs = {"source": ds.attrs.get("source", ""),
                 "geometry": repr(options.geometry),
                 "options": repr(options)}
    return out


GROUPS = {
    "hflux": ("HAKE", "HCV"),
    "vflux": ("VAKE", "VCV"),
    "pe": ("PEKE",),
    "cond": ("POV", "TOV", "COV"),
    "fric": ("MOKE", "TOKE", "GOKE"),
    "ana": ("AOKEV",),
}


def groups(result, quantity="KEV"):
    """The budget of `quantity` in groups, as in the reference's
    make_website_budget.py (kg s-3):

        tend  = I{E}TEND, the change of the instantaneous column over the window
        hflux = IHAKE + IHCV = -<div(E u)>
        vflux = IVAKE + IVCV = -<d(E omega)/dp>
        pe    = IPEKE
        cond  = IPOV + ITOV + ICOV, all the vapor-side physics
        fric  = IMOKE + ITOKE + IGOKE, all the momentum-side physics
        ana   = IAOKEV
        spte  = SPTE
        resid = tend - (the seven groups above)

    "cond" and "fric" are the reference's names; cond includes turbulence
    (surface evaporation and mixing) and fric includes moist processes
    (convective momentum transport). Terms missing from `result` (e.g. no
    physics in ERA5) count as zero, so the residual absorbs them.
    """
    def column(term):
        name = f"I{_name(term, quantity)}"
        return result[name] if name in result else 0.0

    out = {"tend": result[f"I{quantity}TEND"]}
    for group, members in GROUPS.items():
        out[group] = sum(column(t) for t in members)
    out["spte"] = result[_name("SPTE", quantity)]
    out["resid"] = out["tend"] - sum(out[g] for g in list(GROUPS) + ["spte"])
    ds = xr.Dataset({g: v if isinstance(v, xr.DataArray) else xr.zeros_like(out["tend"])
                     for g, v in out.items()})
    for v in ds.data_vars.values():
        v.attrs = {"units": "kg s-3"}
    return ds
