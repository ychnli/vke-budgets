"""
Output variable names, their budget groups, and the translation to the
some legacy names.

Every budget variable is named <quantity>_<term>, with quantity `kev`
(qK) or `vte` (q^2 K), and term named after its group:

    kev_hflux_ke        member "ke" of group "hflux"
    kev_hflux           (the group itself, from budget.groups)

Column integrals (the default output, kg s-3 for tendency terms) carry the
plain name; the same term on pressure levels (keep_3d, m2 s-3) adds `_3d`.

The budget of E = q^n K, in groups (budget.groups):

    tend = hflux + vflux + pe_conversion + subgrid_momentum
           + net_vapor_source + analysis + boundary + residual

The reference called subgrid_momentum "fric" and net_vapor_source "cond".
Neither name was accurate: subgrid_momentum is every parametrized force on
the wind (turbulence, convection, gravity-wave drag), and net_vapor_source
every source and sink of vapor (condensation, evaporation, turbulent
transport from the surface, chemistry).

The translation to the reference's names (IHAKE, IHAKE_v, SPTE, ...) is
kept here, because the legacy file layout (io.write_legacy, io.open_legacy)
uses it. docs/names.md lists it as a table (`python -m vke_budgets.names`).
"""
from dataclasses import dataclass

QUANTITY_PREFIX = {"KEV": "kev", "VTE": "vte"}


@dataclass(frozen=True)
class Term:
    """A term of the budget of E = q^n K.

    Attributes:
        long_name: description; "{Q}" is replaced by the quantity (KEV, VTE).
        legacy: the reference's name of the term in the KEV budget, without
            the "I" of the column integral ("HAKE" for IHAKE); None if the
            reference has no such term.
        group: the budget group it belongs to, or None for diagnostics.
        units: (column units, 3D units).
        quantities: the budgets that have the term.
    """
    long_name: str
    legacy: str | None
    group: str | None = None
    units: tuple = ("kg s-3", "m2 s-3")
    quantities: tuple = ("KEV", "VTE")


TERMS = {
    # --- The budget, group by group ---------------------------------------
    "tend": Term("change of the column {Q} over the window, (<E>(end) - <E>(start))/dt", None, "tend"),
    # hflux
    "hflux_ke": Term("horizontal advection of K, -q^n u.grad(K)", "HAKE", "hflux"),
    "hflux_vapor": Term("horizontal convergence of the q^n transport, -K div(q^n u)", "HCV", "hflux"),
    # vflux
    "vflux_ke": Term("vertical advection of K, -q^n omega u.du/dp", "VAKE", "vflux"),
    "vflux_vapor": Term("vertical convergence of the q^n transport, -K d(q^n omega)/dp", "VCV", "vflux"),
    # pe conversion
    "pe_conversion": Term("conversion of potential energy to K, -q^n u.grad(Phi)", "PEKE", "pe_conversion"),
    # subgrid-scale momentum forcing
    "subgrid_momentum_moist": Term("subgrid momentum forcing by moist processes (convective momentum transport)", "MOKE", "subgrid_momentum"),
    "subgrid_momentum_turb": Term("subgrid momentum forcing by turbulence", "TOKE", "subgrid_momentum"),
    "subgrid_momentum_gwd": Term("subgrid momentum forcing by gravity-wave drag", "GOKE", "subgrid_momentum"),
    # vapor sources and sinks 
    "net_vapor_source_moist": Term("net vapor source from moist processes (condensation, re-evaporation)", "POV", "net_vapor_source"),
    "net_vapor_source_turb": Term("net vapor source from turbulence (surface evaporation, mixing)", "TOV", "net_vapor_source"),
    "net_vapor_source_chem": Term("net vapor source from chemistry", "COV", "net_vapor_source"),
    # analysis increment
    "analysis": Term("analysis increments of wind and vapor", "AOKEV", "analysis"),
    
    # boundary term
    "boundary": Term("moving lower boundary, (dps/dt) E(p_bottom)/g", "SPTE", "boundary"),

    # --- Diagnostics ------------------------------------------------------
    "tend_local": Term("integral of the local tendency, q^n u.du/dt + n q^(n-1) K dq/dt",
                       None),
    "mean": Term("column {Q} of the window-mean state", None, units=("kg s-2", "m2 s-2")),
    "hadv_vapor": Term("horizontal advection of q, -n q^(n-1) K u.grad(q) "
                       "(advective form of hflux_vapor)", "HAV"),
    "vadv_vapor": Term("vertical advection of q, -n q^(n-1) K omega dq/dp "
                       "(advective form of vflux_vapor)", "VAV"),
    "model_dynamics_momentum": Term("the model's own dynamics tendency of the wind",
                                    "DOKE"),
    "model_dynamics_vapor": Term("the model's own dynamics tendency of vapor", "DOV"),
    "hflux_vapor_ncl": Term("horizontal convergence, -2qK div(q u) "
                            "(form commented out in the NCL)", None, quantities=("VTE",)),
    "vflux_vapor_ncl": Term("vertical convergence, -2qK d(q omega)/dp "
                            "(form commented out in the NCL)", None, quantities=("VTE",)),
    # --- Flux-form decomposition (KEV only) --------------------------------
    "fluxform_hflux_ke": Term("horizontal flux of K, -q div(K u)", "HFOKE",
                              quantities=("KEV",)),
    "fluxform_vflux_ke": Term("vertical flux of K, -q d(K omega)/dp", "VFOKE",
                              quantities=("KEV",)),
    "fluxform_hflux_vapor": Term("horizontal flux of vapor, -K div(q u) (= hflux_vapor)",
                                 "HFOV", quantities=("KEV",)),
    "fluxform_vflux_vapor": Term("vertical flux of vapor, -K d(q omega)/dp (= vflux_vapor)",
                                 "VFOV", quantities=("KEV",)),
    "fluxform_hcross": Term("horizontal divergence cross term, q K div(u)", "HFADT",
                            quantities=("KEV",)),
    "fluxform_vcross": Term("vertical divergence cross term, q K d(omega)/dp", "VFADT",
                            quantities=("KEV",)),
    "continuity_error": Term("2 q K times the 3D divergence (zero by continuity)", "QKDIVV",
                             quantities=("KEV",)),
    "pe_baroclinic": Term("baroclinic conversion, -omega alpha q", "BARCC",
                          quantities=("KEV",)),
    "pe_hflux_geopotential": Term("horizontal flux of geopotential, -q div(Phi u)",
                                  "HFOGEOP", quantities=("KEV",)),
    "pe_flux_geopotential": Term("3D flux of geopotential", "FOGEOP", quantities=("KEV",)),
    "pe_continuity_error": Term("q Phi times the 3D divergence (zero by continuity)",
                                "QPHIDIVV", quantities=("KEV",)),
}

# Variables that belong to no single budget: new name -> (legacy name,
# long name, units). The *_3d entries exist only with keep_3d.
SHARED = {
    "ivt": ("IVT", "integrated vapor transport, the magnitude of <q u>", "kg m-1 s-1"),
    "kev": ("IKEV", "column KEV <q K>, instantaneous", "kg s-2"),
    "vte": ("IVTE", "column VTE <q^2 K>, instantaneous", "kg s-2"),
    "divergence_3d_error": ("IDIVV", "column 3D divergence, <div(u) + d(omega)/dp> "
                            "(zero by continuity)", "kg m-2 s-1"),
    "divergence_3d_error_3d": ("DIVV", "3D divergence, div(u) + d(omega)/dp "
                               "(zero by continuity)", "s-1"),
    "specific_volume_3d": ("ALPHA", "specific volume, -dPhi/dp", "m3 kg-1"),
}

# The groups of budget.groups, in budget order.
GROUPS = ["tend", "hflux", "vflux", "pe_conversion", "subgrid_momentum",
          "net_vapor_source", "analysis", "boundary"]
GROUP_LONG_NAMES = {
    "tend": "change of the column over the window",
    "hflux": "horizontal flux divergence, -<div(E u)>",
    "vflux": "vertical flux divergence, -<d(E omega)/dp>",
    "pe_conversion": "conversion of potential energy to K",
    "subgrid_momentum": "subgrid-scale momentum forcing (turbulence, convection, "
                        "gravity-wave drag)",
    "net_vapor_source": "net vapor source, all sources and sinks of vapor",
    "analysis": "analysis increments",
    "boundary": "moving lower boundary (surface pressure tendency)",
    "residual": "tend minus every group",
}


def name(term, quantity, three_d=False):
    """Output name of `term` in the budget of `quantity` ("KEV" or "VTE")."""
    return f"{QUANTITY_PREFIX[quantity]}_{term}" + ("_3d" if three_d else "")


def attrs(term, quantity, three_d=False):
    """units, long_name, budget group and legacy name of a budget variable."""
    t = TERMS[term]
    out = {"units": t.units[1] if three_d else t.units[0],
           "long_name": f"{quantity} budget: " + t.long_name.replace("{Q}", quantity)}
    if t.group is not None:
        out["budget_group"] = t.group
    legacy = _legacy_budget_name(term, quantity, three_d)
    if legacy is not None:
        out["legacy_name"] = legacy
    return out


def members(group):
    """The terms of a budget group, in table order."""
    return [term for term, t in TERMS.items() if t.group == group]


def _legacy_budget_name(term, quantity, three_d):
    """The reference's name for a budget variable, or None."""
    vte = quantity == "VTE"
    q = quantity
    special = {
        "tend": (f"I{q}TEND", None),
        "tend_local": (f"I_{q}TEND", f"{q}TEND"),
        "mean": (f"I{q}AVG", q),
        "boundary": ("SPTE_v" if vte else "SPTE", None),
        "hflux_vapor_ncl": ("IHCV_vO", "HCV_vO"),
        "vflux_vapor_ncl": ("IVCV_vO", "VCV_vO"),
    }
    if term in special:
        return special[term][1 if three_d else 0]
    base = TERMS[term].legacy
    if base is None:
        return None
    base = base + ("_v" if vte else "")
    return base if three_d else f"I{base}"


def legacy_names():
    """{new name: legacy name} for every variable that has one."""
    table = {}
    for term, t in TERMS.items():
        for quantity in t.quantities:
            for three_d in (False, True):
                legacy = _legacy_budget_name(term, quantity, three_d)
                if legacy is not None:
                    table[name(term, quantity, three_d)] = legacy
    table.update({new: legacy for new, (legacy, _, _) in SHARED.items()})
    return table


def to_legacy(ds):
    """Rename a result's variables to the reference's names."""
    table = legacy_names()
    return ds.rename({v: table[v] for v in ds.data_vars if v in table})


def from_legacy(ds):
    """Rename a reference-named Dataset's variables to the package's names."""
    table = {legacy: new for new, legacy in legacy_names().items()}
    return ds.rename({v: table[v] for v in ds.data_vars if v in table})


def translation_table():
    """docs/names.md: every variable, its group, and its reference name."""
    lines = ["# Variable names", "",
             "Generated by `python -m vke_budgets.names > docs/names.md`; see",
             "`src/vke_budgets/names.py`. Column integrals are listed; the same term on",
             "pressure levels (`keep_3d`) adds `_3d`, and its reference name drops the `I`.",
             "", "| name (KEV) | name (VTE) | group | reference KEV | reference VTE | meaning |",
             "|---|---|---|---|---|---|"]
    for term, t in TERMS.items():
        kev = name(term, "KEV") if "KEV" in t.quantities else ""
        vte = name(term, "VTE") if "VTE" in t.quantities else ""
        ref_kev = _legacy_budget_name(term, "KEV", False) if kev else ""
        ref_vte = _legacy_budget_name(term, "VTE", False) if vte else ""
        lines.append(f"| `{kev}` | `{vte}` | {t.group or 'diagnostic'} | {ref_kev or ''} | "
                     f"{ref_vte or ''} | {t.long_name.replace('{Q}', 'E')} |"
                     .replace("| `` |", "|  |"))
    lines += ["", "Variables that belong to no single budget:", "",
              "| name | reference | meaning |", "|---|---|---|"]
    lines += [f"| `{new}` | {legacy} | {long_name} |"
              for new, (legacy, long_name, _) in SHARED.items()]
    lines += ["", "Groups (`budget.groups`), in budget order:", "",
              "| group | reference | meaning |", "|---|---|---|"]
    old = {"pe_conversion": "pe", "subgrid_momentum": "fric", "net_vapor_source": "cond",
           "analysis": "ana", "boundary": "spte", "residual": "resid"}
    lines += [f"| `{g}` | {old.get(g, g)} | {GROUP_LONG_NAMES[g]} |"
              for g in GROUPS + ["residual"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(translation_table(), end="")
