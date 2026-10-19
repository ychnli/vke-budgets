"""
Assemble the KEV and VTE budgets from a standard input Dataset
(standard.py) one averaging window at a time.

For each window:

1. On model levels, form the vertical-derivative pieces omega du/dp,
   omega dv/dp, omega dq/dp and d(q^n omega)/dp. Model levels never go below
   the ground, and their spacing is finest near the surface.
2. Interpolate the window-mean state and those pieces to pressure levels.
3. On pressure levels, form every term of terms.py, taking horizontal
   derivatives along pressure surfaces.
4. Integrate each term over the column, p_bottom to p_top (1000-200 hPa).
5. Add the tendencies from the instantaneous fields at the two ends of the
   window: tend = (<E>(end) - <E>(start)) / dt, and the boundary term.

The budget closes as (see groups and names.py)

    tend = hflux + vflux + pe_conversion + subgrid_momentum
           + net_vapor_source + analysis + boundary + residual.

Variables are named <quantity>_<term> (kev_hflux_ke, vte_hflux_ke, ...);
names.py has every name, its group, and the reference's name for it.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import xarray as xr

from . import names, terms
from .geometry import DEFAULT, HorizontalGeometry
from .standard import check_input, physics_processes
from .vertical import GRAVITY, PressureInterpolator, bottom_value, ddp, integrate_column

# Budgeted quantities E = q^n K: name -> n.
QUANTITIES = {"KEV": 1, "VTE": 2}
QUANTITY_PREFIX = names.QUANTITY_PREFIX

# Physics tendencies -> terms (see names.py).
MOMENTUM_TERMS = {"moist": "subgrid_momentum_moist", "turb": "subgrid_momentum_turb",
                  "gwd": "subgrid_momentum_gwd", "dyn": "model_dynamics_momentum"}
MOISTURE_TERMS = {"moist": "net_vapor_source_moist", "turb": "net_vapor_source_turb",
                  "chem": "net_vapor_source_chem", "dyn": "model_dynamics_vapor"}


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
        dyn_crosscheck: also compute the terms of the model's own dynamics
            tendencies of the wind (model_dynamics_momentum, DOKE in the
            reference) and of vapor (model_dynamics_vapor, DOV), a
            cross-check on hflux + vflux + pe_conversion.
        ncl_vte_form: also compute vte_hflux_vapor_ncl and
            vte_vflux_vapor_ncl, the form commented out in the NCL.
        flux_form: also compute the flux-form decomposition of the KEV
            budget (kev_fluxform_* and the continuity diagnostics, see
            terms.py).
        boundary_everywhere: apply the boundary term (SPTE) at every point,
            as the reference does. By default it is zero where the surface
            pressure stays above p_bottom over the window, because there the
            bottom of the integrated column is fixed at p_bottom and does
            not move (see terms.surface_pressure_term).
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
    flux_form: bool = False
    boundary_everywhere: bool = False


def instantaneous_columns(inst, options=BudgetOptions()):
    """Column integrals of the instantaneous state (dims time_inst, lat, lon):
    ivt = |<q u>| (kg m-1 s-1), and kev = <q K>, vte = <q^2 K> (kg s-2)."""
    integ = _integrator(options)
    q, u, v = inst["q_inst"], inst["u_inst"], inst["v_inst"]
    k = terms.kinetic_energy(u, v)
    out = {"ivt": np.hypot(integ(q * u), integ(q * v)).assign_attrs(_shared_attrs("ivt"))}
    for quantity in options.quantities:
        column = QUANTITY_PREFIX[quantity]
        out[column] = integ(q ** QUANTITIES[quantity] * k).assign_attrs(_shared_attrs(column))
    return xr.Dataset(out)


def _shared_attrs(new):
    legacy, long_name, units = names.SHARED[new]
    return {"units": units, "long_name": long_name, "legacy_name": legacy}


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
    if options.flux_form:
        k_ml, phi_ml = terms.kinetic_energy(u_ml, v_ml), window["phi"]
        pieces.update({
            "d_komega_dp": ddp(k_ml * omega_ml, p, "mlev"),
            "d_omega_dp": ddp(omega_ml, p, "mlev"),
            "d_phi_dp": ddp(phi_ml, p, "mlev"),
            "d_omegaphi_dp": ddp(omega_ml * phi_ml, p, "mlev"),
            "omega": omega_ml,
        })
        if "d_q1omega_dp" not in pieces:
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
    # The integrated column starts at min(p_s, p_bottom): its bottom can only
    # move where p_s is below p_bottom at either end of the window.
    bottom_moves = ((inst_start["ps_inst"] < options.p_bottom)
                    | (inst_end["ps_inst"] < options.p_bottom) | options.boundary_everywhere)

    momentum, moisture = physics_processes(window)
    out = {}

    def add(term, quantity, field):
        """The column integral of a term, and the term itself with keep_3d."""
        out[names.name(term, quantity)] = integ(field).assign_attrs(names.attrs(term, quantity))
        if options.keep_3d:
            out[names.name(term, quantity, True)] = field.assign_attrs(
                names.attrs(term, quantity, True))

    for quantity in options.quantities:
        n = QUANTITIES[quantity]
        w_m, w_v = terms.weights(q, k, n)
        e = w_m * k

        # 3. Terms on pressure levels (m2 s-3), by group.
        fields = {
            "hflux_ke": terms.horizontal_advection_of_ke(w_m, u, v, k, geom),
            "hflux_vapor": terms.horizontal_convergence_of_vapor(k, w_m, u, v, geom),
            "vflux_ke": terms.vertical_advection_of_ke(w_m, u, v, pieces["omega_dudp"],
                                                       pieces["omega_dvdp"]),
            "vflux_vapor": terms.vertical_convergence_of_vapor(k, pieces[f"d_q{n}omega_dp"]),
            "pe_conversion": terms.pe_to_ke_conversion(w_m, u, v, phi, geom),
        }
        for proc, term in MOMENTUM_TERMS.items():
            if proc in momentum and (proc != "dyn" or options.dyn_crosscheck):
                fields[term] = terms.momentum_physics(w_m, u, v, window[f"dudt_{proc}"],
                                                      window[f"dvdt_{proc}"])
        for proc, term in MOISTURE_TERMS.items():
            if proc in moisture and (proc != "dyn" or options.dyn_crosscheck):
                fields[term] = terms.moisture_physics(w_v, window[f"dqdt_{proc}"])
        if "ana" in momentum and "ana" in moisture:
            fields["analysis"] = (terms.momentum_physics(w_m, u, v, window["dudt_ana"],
                                                         window["dvdt_ana"])
                                  + terms.moisture_physics(w_v, window["dqdt_ana"]))
        # Diagnostics: the advective form of the vapor terms, and the NCL form.
        fields["hadv_vapor"] = terms.horizontal_advection_of_vapor(w_v, u, v, q, geom)
        fields["vadv_vapor"] = terms.vertical_advection_of_vapor(w_v, pieces["omega_dqdp"])
        if options.ncl_vte_form and n == 2:
            fields["hflux_vapor_ncl"], fields["vflux_vapor_ncl"] = \
                terms.ncl_convergence_of_vapor(w_v, q, u, v, pieces["d_q1omega_dp"], geom)
        fields["tend_local"] = terms.local_tendency(w_m, w_v, u, v, dqdt, dudt, dvdt)
        fields["mean"] = e

        # 4. Column integrals (kg s-3; kg s-2 for the mean).
        for term, field in fields.items():
            add(term, quantity, field)

        # 5. Change of the instantaneous column over the window, and the
        #    moving lower boundary.
        column = QUANTITY_PREFIX[quantity]
        out[names.name("tend", quantity)] = (
            (inst_end[column] - inst_start[column]) / dt).assign_attrs(
            names.attrs("tend", quantity))
        out[names.name("boundary", quantity)] = terms.surface_pressure_term(
            dpsdt, bottom_value(e, dim="lev"), options.g, bottom_moves).assign_attrs(
            names.attrs("boundary", quantity))

    if options.flux_form:
        for term, field in _flux_form(q, k, u, v, phi, pieces, geom).items():
            if term in names.TERMS:
                add(term, "KEV", field)
        divv = terms.three_d_divergence(u, v, pieces["d_omega_dp"], geom)
        out["divergence_3d_error"] = integ(divv).assign_attrs(
            _shared_attrs("divergence_3d_error"))
        if options.keep_3d:
            out["divergence_3d_error_3d"] = divv.assign_attrs(
                _shared_attrs("divergence_3d_error_3d"))
            out["specific_volume_3d"] = (-pieces["d_phi_dp"]).assign_attrs(
                _shared_attrs("specific_volume_3d"))

    result = xr.Dataset(out)
    if options.pole_lat_limit is not None:
        result = result.where(np.abs(result["lat"]) < options.pole_lat_limit)
    return result


def _flux_form(q, k, u, v, phi, pieces, geom):
    """The flux-form decomposition of the KEV budget on pressure levels
    (see terms.py), by term name."""
    divv = terms.three_d_divergence(u, v, pieces["d_omega_dp"], geom)
    _, barcc, hfogeop, fogeop, qphidivv = terms.geopotential_flux_terms(
        q, phi, u, v, pieces["omega"], pieces["d_phi_dp"], pieces["d_omegaphi_dp"], divv, geom)
    return {
        "fluxform_hflux_ke": terms.horizontal_flux_of_ke(q, k, u, v, geom),
        "fluxform_vflux_ke": terms.vertical_flux_of_ke(q, pieces["d_komega_dp"]),
        "fluxform_hflux_vapor": terms.horizontal_convergence_of_vapor(k, q, u, v, geom),
        "fluxform_vflux_vapor": terms.vertical_convergence_of_vapor(k, pieces["d_q1omega_dp"]),
        "fluxform_hcross": terms.horizontal_divergence_cross_term(q, k, u, v, geom),
        "fluxform_vcross": terms.vertical_divergence_cross_term(q, k, pieces["d_omega_dp"]),
        "continuity_error": 2.0 * q * k * divv,
        "pe_baroclinic": barcc,
        "pe_hflux_geopotential": hfogeop,
        "pe_flux_geopotential": fogeop,
        "pe_continuity_error": qphidivv,
    }


INST_INPUTS = ["q_inst", "u_inst", "v_inst", "ps_inst"]


def iter_windows(ds, options=BudgetOptions(), windows=None, n_workers=1, progress=False):
    """Yield the budget of each window in turn, as compute() would return it
    for that window alone (window terms on a length-1 `time`, and the
    instantaneous columns at its two ends on `time_inst`).

    Use it to stream: reduce each window (a sum, a composite, a regional
    mean) and let it go, without holding a whole period in memory. With
    n_workers > 1, up to n_workers windows are computed ahead in threads.
    """
    check_input(ds)
    windows = range(ds.sizes["time"]) if windows is None else list(windows)
    seconds = (ds["time_inst"].values[1:] - ds["time_inst"].values[:-1]) / np.timedelta64(1, "s")
    mean_state = ds.drop_dims("time_inst")

    def one(i):
        inst = ds[INST_INPUTS].isel(time_inst=[i, i + 1]).load()
        inst = inst.merge(instantaneous_columns(inst, options))
        window = mean_state.isel(time=i).load()
        result = window_budget(window, inst.isel(time_inst=0), inst.isel(time_inst=1),
                               seconds[i], options)
        result = result.drop_vars(["time", "time_inst"], errors="ignore")
        result = result.expand_dims(time=[window["time"].values])
        columns = inst[["ivt"] + [QUANTITY_PREFIX[qn] for qn in options.quantities]]
        if progress:
            print(f"  window {str(window['time'].values)[:16]}", flush=True)
        return xr.merge([result, columns])

    if n_workers <= 1:
        for i in windows:
            yield one(i)
        return
    with ThreadPoolExecutor(n_workers) as pool:
        # Keep n_workers windows in flight, in order.
        pending = [pool.submit(one, i) for i in windows[:n_workers]]
        for i in windows[n_workers:]:
            yield pending.pop(0).result()
            pending.append(pool.submit(one, i))
        for future in pending:
            yield future.result()


def combine(parts):
    """Join per-window (or per-day) results in time: the window terms along
    `time`, the instantaneous columns along `time_inst`, each instantaneous
    step once."""
    windows = xr.concat([p.drop_dims("time_inst") for p in parts], dim="time")
    inst = xr.concat([p[[v for v in p.data_vars if "time_inst" in p[v].dims]] for p in parts],
                     dim="time_inst").drop_duplicates("time_inst")
    return xr.merge([windows, inst])


def compute(ds, options=BudgetOptions(), windows=None, n_workers=1, progress=False):
    """The budgets for the windows of a standard input Dataset.

    Args:
        ds: standard input Dataset (e.g. sources.merra2.open_day), lazy or
            in memory.
        options: BudgetOptions.
        windows: indices of the windows to compute (default: all).
        n_workers: windows computed at once, in threads. Each window needs
            about 3 GB.
        progress: print one line per window.
    Returns:
        Dataset with the window terms on `time` and the instantaneous column
        integrals (ivt, kev, vte) on `time_inst`, for the selected windows
        and the instantaneous steps that bound them.
    """
    parts = list(iter_windows(ds, options, windows, n_workers, progress))
    out = combine(parts)
    out["time"].attrs = {"long_name": "centre of the averaging window"}
    out["time_inst"].attrs = {"long_name": "instantaneous time (the ends of the windows)"}
    out.attrs = {"source": ds.attrs.get("source", ""),
                 "options": repr(options)}
    return out


def groups(result, quantity="KEV"):
    """The budget of `quantity` in groups (kg s-3), in budget order:

        tend             change of the column over the window
        hflux            hflux_ke + hflux_vapor = -<div(E u)>
        vflux            vflux_ke + vflux_vapor = -<d(E omega)/dp>
        pe_conversion    conversion of potential energy to K
        subgrid_momentum subgrid-scale momentum forcing: turbulence,
                         convection, gravity-wave drag ("fric" in the reference)
        net_vapor_source every source and sink of vapor: moist processes,
                         turbulence, chemistry ("cond" in the reference)
        analysis         analysis increments
        boundary         moving lower boundary
        residual         tend minus all of the above

    Groups whose terms are missing from `result` (e.g. no physics in ERA5)
    count as zero, so the residual absorbs them.
    """
    tend = result[names.name("tend", quantity)]
    out = {}
    for group in names.GROUPS:
        present = [result[names.name(t, quantity)] for t in names.members(group)
                   if names.name(t, quantity) in result]
        out[group] = sum(present) if present else xr.zeros_like(tend)
    out["residual"] = tend - sum(out[g] for g in names.GROUPS if g != "tend")
    ds = xr.Dataset(out)
    for group, da in ds.data_vars.items():
        da.attrs = {"units": "kg s-3", "long_name": f"{quantity} budget: "
                    f"{names.GROUP_LONG_NAMES[group]}"}
    n = QUANTITIES[quantity]
    ds.attrs = {
        "title": f"Column {quantity} budget by group",
        "quantity": f"{quantity} = q^{n} K" if n > 1 else f"{quantity} = q K",
        "budget": "tend = " + " + ".join(g for g in names.GROUPS if g != "tend") + " + residual",
        "column": "mass-weighted integral from 1000 to 200 hPa",
        "source": result.attrs.get("source", ""),
    }
    return ds
