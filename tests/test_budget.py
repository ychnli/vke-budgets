"""
Budget assembly on synthetic standard input with known answers: solid-body
rotation with uniform humidity, where every dynamical term vanishes, plus
simple physics tendencies whose terms are known in closed form.
"""
import numpy as np
import xarray as xr

from synthetic import COLUMN_MASS, DT, LAT, Q0, U0, kinetic_energy, solid_body
from vke_budgets.vertical import GRAVITY
from vke_budgets import budget

INNER = np.abs(LAT) < 87


def assert_close(actual, expected, **kw):
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), **kw)


def test_dynamical_terms_vanish_for_solid_body_rotation():
    out = budget.compute(solid_body())
    scale = 2 * Q0 * kinetic_energy().max() * COLUMN_MASS / DT   # a generous term size
    for quantity in ("kev", "vte"):
        for term in ("hflux_ke", "hflux_vapor", "vflux_ke", "vflux_vapor", "pe_conversion",
                     "hadv_vapor", "vadv_vapor"):
            name = f"{quantity}_{term}"
            np.testing.assert_allclose(out[name].values[:, INNER], 0.0, atol=1e-12 * scale,
                                       err_msg=name)


def test_column_quantities():
    out = budget.compute(solid_body())
    k = kinetic_energy()
    assert_close(out["kev_mean"].values[:, INNER], (Q0 * k * COLUMN_MASS)[INNER], rtol=1e-6)
    assert_close(out["vte_mean"].values[:, INNER], (Q0**2 * k * COLUMN_MASS)[INNER], rtol=1e-6)
    assert_close(out["ivt"].values, Q0 * U0 * np.cos(np.deg2rad(LAT))[:, None] * COLUMN_MASS,
                 rtol=1e-6)
    # Window outputs are blanked at the poles; instantaneous ones are not.
    assert np.isnan(out["kev_mean"].values[:, ~INNER]).all()
    assert np.isfinite(out["kev"].values).all()


def test_physics_terms_have_their_closed_form():
    """Drag du/dt = -r u gives subgrid_momentum_turb = q u.(-r u) = -2 r q K,
    and for VTE -2 r q^2 K. A sink dq/dt = -c gives net_vapor_source_moist
    = -c K, and for VTE -2 c q K."""
    r, c = 1e-5, 1e-8
    out = budget.compute(solid_body(drag=r, drying=c))
    k = kinetic_energy()[INNER]
    expected = {"kev_subgrid_momentum_turb": -2 * r * Q0 * k,
                "vte_subgrid_momentum_turb": -2 * r * Q0**2 * k,
                "kev_net_vapor_source_moist": -c * k,
                "vte_net_vapor_source_moist": -2 * c * Q0 * k}
    for name, value in expected.items():
        assert_close(out[name].values[:, INNER], value * COLUMN_MASS, rtol=1e-6, err_msg=name)
    # Only the physics present in the input is computed.
    assert "kev_subgrid_momentum_moist" not in out and "kev_analysis" not in out


def test_tendency_of_moistening_column():
    """q_inst grows by a factor f over each window: kev_tend = (f - 1) <q K>/dt
    for the start of the window, and the local tendency integrates to the same
    thing with the window-mean q."""
    f = 1.1
    out = budget.compute(solid_body(moistening_inst=f))
    k = kinetic_energy()[INNER]
    start = Q0 * f ** np.arange(2)[:, None, None]
    np.testing.assert_allclose(out["kev_tend"].values[:, INNER],
                               (f - 1) * start * k * COLUMN_MASS / DT, rtol=1e-5)
    np.testing.assert_allclose(out["kev_tend_local"].values[:, INNER],
                               k * (f - 1) * start * COLUMN_MASS / DT, rtol=1e-5)
    # The surface pressure does not change.
    np.testing.assert_array_equal(out["kev_boundary"].values[:, INNER], 0.0)


def test_window_selection_and_options():
    ds = solid_body(n_time=3)
    out = budget.compute(ds, windows=[1])
    assert out.sizes["time"] == 1 and out.sizes["time_inst"] == 2
    np.testing.assert_array_equal(out["time"], ds["time"][1:2])
    np.testing.assert_array_equal(out["time_inst"], ds["time_inst"][1:3])

    out = budget.compute(ds, budget.BudgetOptions(quantities=("KEV",), keep_3d=True), windows=[0])
    assert "vte_hflux_ke" not in out and "vte" not in out
    assert out["kev_hflux_ke_3d"].dims == ("time", "lev", "lat", "lon")
    assert out["kev_hflux_ke_3d"].attrs["units"] == "m2 s-3"
    assert out["kev_hflux_ke"].attrs == {
        "units": "kg s-3", "budget_group": "hflux", "legacy_name": "IHAKE",
        "long_name": "KEV budget: horizontal advection of K, -q^n u.grad(K)"}


def test_threads_give_the_same_answer():
    ds = solid_body(n_time=3, drag=1e-5)
    xr.testing.assert_identical(budget.compute(ds), budget.compute(ds, n_workers=3))


def test_groups_and_residual():
    """With drag only, subgrid_momentum = the turbulence term, every other
    group is 0, tend = 0, so residual = -subgrid_momentum. Groups missing
    from the input count as zero."""
    out = budget.compute(solid_body(drag=1e-5))
    for quantity, prefix in (("KEV", "kev"), ("VTE", "vte")):
        g = budget.groups(out, quantity)
        assert list(g.data_vars) == ["tend", "hflux", "vflux", "pe_conversion",
                                     "subgrid_momentum", "net_vapor_source", "analysis",
                                     "boundary", "residual"]
        turb = out[f"{prefix}_subgrid_momentum_turb"]
        xr.testing.assert_allclose(g.subgrid_momentum, turb)
        xr.testing.assert_allclose(g.residual,
                                   g.tend - sum(g[k] for k in list(g.data_vars)[1:-1]))
        np.testing.assert_allclose(g.residual.values[:, INNER], -turb.values[:, INNER],
                                   rtol=1e-6, atol=1e-12)
        assert float(np.abs(g.net_vapor_source).max()) == 0.0


def test_flux_form_matches_advective_form_for_solid_body_rotation():
    """Every flux-form term vanishes too, and DIVV is zero (the flow is
    nondivergent); the geopotential decomposition needs only alpha = -dPhi/dp."""
    out = budget.compute(solid_body(), budget.BudgetOptions(flux_form=True))
    for name in ("kev_fluxform_hflux_ke", "kev_fluxform_vflux_ke", "kev_fluxform_hflux_vapor",
                 "kev_fluxform_vflux_vapor", "kev_fluxform_hcross", "kev_fluxform_vcross",
                 "divergence_3d_error", "kev_continuity_error", "kev_pe_baroclinic",
                 "kev_pe_continuity_error"):
        np.testing.assert_allclose(out[name].values[:, INNER], 0.0, atol=1e-12, err_msg=name)
    # No vertical motion: the 3D flux of geopotential is the horizontal one.
    xr.testing.assert_allclose(out["kev_pe_flux_geopotential"], out["kev_pe_hflux_geopotential"])


def test_boundary_term_only_where_the_column_bottom_moves():
    """kev_boundary = (dps/dt) E(bottom)/g where p_s is below 1000 hPa at an
    end of the window, and 0 where it stays above (the bottom is fixed at
    1000 hPa); boundary_everywhere applies it everywhere, as the reference."""
    rate = 100.0                                   # Pa per window
    e_bottom = Q0 * kinetic_energy()[INNER]        # E at 1000 hPa, uniform in height
    expected = rate / DT * e_bottom / GRAVITY
    # p_s over the two windows: above 1000 hPa throughout; below throughout;
    # crossing 1000 hPa in the first window (999.5 -> 1000.5 hPa), then above.
    for ps0, moves in ((1013e2, (False, False)), (985e2, (True, True)),
                       (999.5e2, (True, False))):
        ds = solid_body(ps=(ps0, rate))
        out = budget.compute(ds)["kev_boundary"].values[:, INNER]
        ref = budget.compute(ds, budget.BudgetOptions(boundary_everywhere=True))
        assert_close(ref["kev_boundary"].values[:, INNER], expected, rtol=1e-6)
        for window, window_moves in enumerate(moves):
            assert_close(out[window], expected if window_moves else 0.0, rtol=1e-6, atol=0)
