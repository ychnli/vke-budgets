"""
Budget terms for E = q^n K, the vapor kinetic energies of atmospheric rivers.

    n = 1:  KEV = q K
    n = 2:  VTE = q^2 K

with q the specific humidity and K = (u^2 + v^2)/2 the kinetic energy per
unit mass. All terms are per unit mass, on pressure levels, in m2 s-3 
(strictly kg kg-1 m2 s-3 for n = 1).

The budget equation
-------------------
The product rule gives

    dE/dt = q^n u . du/dt  +  n q^(n-1) K dq/dt

so every momentum term carries the weight w_m = q^n (times u .), and every
vapor term the weight w_v = n q^(n-1) K. For KEV, w_m = q and w_v = K.

Below, each term has its output name (names.py, without the kev_/vte_
prefix) and, in parentheses, the reference's name.

Substituting the momentum equation,

    du/dt = -u . grad(u) - omega du/dp - grad(Phi) - f k x u + F,

and noting that u . (u . grad u) = u . grad K and u . (k x u) = 0 (the
Coriolis force does no work), the momentum side becomes

    hflux_ke (HAKE)      = -w_m u . grad K                 horizontal advection of K
    vflux_ke (VAKE)      = -w_m omega (u du/dp + v dv/dp)  vertical advection of K
    pe_conversion (PEKE) = -w_m u . grad Phi               potential energy to K
    subgrid_momentum_x   =  w_m u . F_x                    parametrized force F_x:
      (MOKE, TOKE, GOKE)                                   moist, turb, gwd

Substituting the moisture equation dq/dt = -u . grad q - omega dq/dp + S,
the vapor side becomes

    hadv_vapor (HAV)     = -w_v u . grad q                 horizontal advection of q
    vadv_vapor (VAV)     = -w_v omega dq/dp                vertical advection of q
    net_vapor_source_x   =  w_v S_x                        source or sink S_x:
      (POV, TOV, COV)                                      moist, turb, chem

Flux form. The budget uses, in place of the advection of q, the convergence
terms

    hflux_vapor (HCV) = -K div(q^n u),   vflux_vapor (VCV) = -K d(q^n omega)/dp

chosen so that hflux_ke + hflux_vapor = -div(E u) and vflux_ke + vflux_vapor
= -d(E omega)/dp exactly: the groups hflux and vflux are the horizontal and
vertical flux divergences of E. Continuity, div(u) + d(omega)/dp = 0, makes
hflux_vapor + vflux_vapor = hadv_vapor + vadv_vapor; on real data the two
pairs differ by the discrete continuity error E (div u + d omega/dp).

For n = 2 the reference also writes hflux_vapor_ncl (HCV_vO) =
-w_v div(q u) and vflux_vapor_ncl (VCV_vO) = -w_v d(q omega)/dp, the form
commented out in the original NCL. Their sum differs from the flux form's by
q^2 K (div u + d omega/dp), which vanishes only by continuity.

The functions below take DataArrays on pressure levels and are pure.
"""
import xarray as xr

from .geometry import HorizontalGeometry
from .horizontal import divergence, gradient


def kinetic_energy(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """K = (u^2 + v^2) / 2 (m2 s-2)."""
    return 0.5 * (u * u + v * v)


def weights(q: xr.DataArray, k: xr.DataArray, n: int) -> tuple[xr.DataArray, xr.DataArray]:
    """(w_m, w_v) = (q^n, n q^(n-1) K): the weights of the momentum and
    vapor tendencies in dE/dt, for E = q^n K."""
    w_m = q**n
    w_v = k if n == 1 else n * q ** (n - 1) * k
    return w_m, w_v


# --- Momentum side --------------------------------------------------------

def horizontal_advection_of_ke(w_m: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                               k: xr.DataArray,
                               geometry: HorizontalGeometry) -> xr.DataArray:
    """hflux_ke (HAKE) = -w_m (u dK/dx + v dK/dy)."""
    dkdx, dkdy = gradient(k, geometry)
    return -w_m * (u * dkdx + v * dkdy)


def vertical_advection_of_ke(w_m: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                             omega_dudp: xr.DataArray,
                             omega_dvdp: xr.DataArray) -> xr.DataArray:
    """vflux_ke (VAKE) = -w_m (u omega du/dp + v omega dv/dp). The products omega du/dp
    are formed on model levels, where the vertical derivative is taken, and
    then interpolated (see budget.py)."""
    return -w_m * (u * omega_dudp + v * omega_dvdp)


def pe_to_ke_conversion(w_m: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                        phi: xr.DataArray, geometry: HorizontalGeometry) -> xr.DataArray:
    """pe_conversion (PEKE) = -w_m (u dPhi/dx + v dPhi/dy): work done by the pressure-gradient
    force, which on pressure surfaces is the gradient of geopotential (phi). It
    converts potential energy into K."""
    dphidx, dphidy = gradient(phi, geometry)
    return -w_m * (u * dphidx + v * dphidy)


def momentum_physics(w_m: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                     dudt: xr.DataArray, dvdt: xr.DataArray) -> xr.DataArray:
    """w_m (u du/dt + v dv/dt) for a physics tendency (du/dt, dv/dt) of the
    wind: subgrid_momentum_moist, _turb, _gwd (MOKE, TOKE, GOKE),
    model_dynamics_momentum (DOKE), and the momentum half of analysis
    (AOKEV)."""
    return w_m * (u * dudt + v * dvdt)


# --- Vapor side -----------------------------------------------------------

def horizontal_advection_of_vapor(w_v: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                                  q: xr.DataArray,
                                  geometry: HorizontalGeometry) -> xr.DataArray:
    """hadv_vapor (HAV) = -w_v (u dq/dx + v dq/dy)."""
    dqdx, dqdy = gradient(q, geometry)
    return -w_v * (u * dqdx + v * dqdy)


def vertical_advection_of_vapor(w_v: xr.DataArray,
                                omega_dqdp: xr.DataArray) -> xr.DataArray:
    """vadv_vapor (VAV) = -w_v omega dq/dp, with omega dq/dp formed on
    model levels."""
    return -w_v * omega_dqdp


def horizontal_convergence_of_vapor(k: xr.DataArray, w_m: xr.DataArray, u: xr.DataArray,
                                    v: xr.DataArray,
                                    geometry: HorizontalGeometry) -> xr.DataArray:
    """hflux_vapor (HCV) = -K div(q^n u), where w_m = q^n. With hflux_ke it
    makes the horizontal flux divergence -div(E u)."""
    return -k * divergence(w_m * u, w_m * v, geometry)


def vertical_convergence_of_vapor(k: xr.DataArray,
                                  d_wm_omega_dp: xr.DataArray) -> xr.DataArray:
    """vflux_vapor (VCV) = -K d(q^n omega)/dp, with the derivative taken on
    model levels. With vflux_ke it makes the vertical flux divergence
    -d(E omega)/dp."""
    return -k * d_wm_omega_dp


def ncl_convergence_of_vapor(w_v: xr.DataArray, q: xr.DataArray, u: xr.DataArray,
                             v: xr.DataArray, d_q_omega_dp: xr.DataArray,
                             geometry: HorizontalGeometry) -> tuple[xr.DataArray, xr.DataArray]:
    """(hflux_vapor_ncl, vflux_vapor_ncl) = (-w_v div(q u), -w_v d(q omega)/dp),
    HCV_vO and VCV_vO in the reference: the form commented out in the
    original NCL, kept for comparison. For n = 1 it equals (hflux_vapor,
    vflux_vapor)."""
    return -w_v * divergence(q * u, q * v, geometry), -w_v * d_q_omega_dp


def moisture_physics(w_v: xr.DataArray, dqdt: xr.DataArray) -> xr.DataArray:
    """w_v dq/dt for a physics tendency of q: net_vapor_source_moist, _turb,
    _chem (POV, TOV, COV), model_dynamics_vapor (DOV), and the vapor half of
    analysis (AOKEV)."""
    return w_v * dqdt


# --- Tendencies -----------------------------------------------------------

def local_tendency(w_m: xr.DataArray, w_v: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                   dqdt: xr.DataArray, dudt: xr.DataArray,
                   dvdt: xr.DataArray) -> xr.DataArray:
    """dE/dt = w_m (u du/dt + v dv/dt) + w_v dq/dt, from the change of the
    instantaneous q, u, v across the window: tend_local (KEVTEND,
    VTETEND)."""
    return w_m * (u * dudt + v * dvdt) + w_v * dqdt


def surface_pressure_term(dpsdt: xr.DataArray, e_bottom: xr.DataArray, g: float,
                          bottom_moves: xr.DataArray | bool = True) -> xr.DataArray:
    """boundary (SPTE) = (dps/dt) E(p_bottom) / g (kg s-3): the Leibniz term
    for a column whose bottom moves with the surface pressure,

        d/dt <E> = <dE/dt> + E(p_bottom) (dp_bottom/dt) / g.

    The second term is the change of the column total because its lower
    edge moves, not because E changes inside it. E(p_bottom) is E at the
    lowest level above ground.

    The integrated column runs from min(p_s, p_bottom) upward, so where the
    surface pressure stays above p_bottom (1000 hPa; most of the ocean) its
    bottom is fixed at p_bottom and the term is zero. bottom_moves is False
    there. The reference applies the term everywhere (bottom_moves=True).
    """
    return (dpsdt * e_bottom / g).where(bottom_moves, 0.0)


# --- Flux-form decomposition (KEV only; diagnostics) -----------------------
#
# The flux divergence of KEV = qK splits by the product rule as
#
#     div3(q K v) = q div3(K v) + K div3(q v) - q K div3(v),
#
# with div3(a v) = div(a u) + d(a omega)/dp. Its horizontal and vertical
# parts, with a minus sign, are
#
#     fluxform_hflux_ke + fluxform_hflux_vapor + fluxform_hcross = -div(q K u)
#     fluxform_vflux_ke + fluxform_vflux_vapor + fluxform_vcross = -d(q K omega)/dp
#
# (HFOKE + HFOV + HFADT and VFOKE + VFOV + VFADT in the reference; HFOV is
# HCV and VFOV is VCV). They equal the groups hflux and vflux analytically,
# but are differenced in a different order. The 3D divergence DIVV =
# div(u) + d(omega)/dp vanishes by continuity, so every term proportional
# to it measures the discrete continuity error.

def horizontal_flux_of_ke(q: xr.DataArray, k: xr.DataArray, u: xr.DataArray,
                          v: xr.DataArray, geometry: HorizontalGeometry) -> xr.DataArray:
    """fluxform_hflux_ke (HFOKE) = -q div(K u)."""
    return -q * divergence(k * u, k * v, geometry)


def vertical_flux_of_ke(q: xr.DataArray, d_k_omega_dp: xr.DataArray) -> xr.DataArray:
    """fluxform_vflux_ke (VFOKE) = -q d(K omega)/dp, the derivative taken on
    model levels."""
    return -q * d_k_omega_dp


def horizontal_divergence_cross_term(q: xr.DataArray, k: xr.DataArray, u: xr.DataArray,
                                     v: xr.DataArray,
                                     geometry: HorizontalGeometry) -> xr.DataArray:
    """fluxform_hcross (HFADT) = q K div(u)."""
    return q * k * divergence(u, v, geometry)


def vertical_divergence_cross_term(q: xr.DataArray, k: xr.DataArray,
                                   d_omega_dp: xr.DataArray) -> xr.DataArray:
    """fluxform_vcross (VFADT) = q K d(omega)/dp."""
    return q * k * d_omega_dp


def three_d_divergence(u: xr.DataArray, v: xr.DataArray, d_omega_dp: xr.DataArray,
                       geometry: HorizontalGeometry) -> xr.DataArray:
    """divergence_3d_error (DIVV) = div(u) + d(omega)/dp (s-1), zero by
    continuity."""
    return divergence(u, v, geometry) + d_omega_dp


def geopotential_flux_terms(q: xr.DataArray, phi: xr.DataArray, u: xr.DataArray,
                            v: xr.DataArray, omega: xr.DataArray, d_phi_dp: xr.DataArray,
                            d_omega_phi_dp: xr.DataArray, divv: xr.DataArray,
                            geometry: HorizontalGeometry) -> tuple[xr.DataArray, ...]:
    """The flux form of pe_conversion = -q u . grad(Phi). With the
    hydrostatic specific volume alpha = -dPhi/dp,

        pe_conversion = pe_flux_geopotential + pe_baroclinic + pe_continuity_error
        pe_hflux_geopotential (HFOGEOP) = -q div(Phi u)
        pe_flux_geopotential (FOGEOP)   = pe_hflux_geopotential - q d(omega Phi)/dp
        pe_baroclinic (BARCC)           = -omega alpha q
        pe_continuity_error (QPHIDIVV)  =  q Phi DIVV

    Returns (alpha, pe_baroclinic, pe_hflux_geopotential,
    pe_flux_geopotential, pe_continuity_error)."""
    alpha = -d_phi_dp
    hfogeop = -q * divergence(phi * u, phi * v, geometry)
    return (alpha, -omega * alpha * q, hfogeop, hfogeop - q * d_omega_phi_dp,
            q * phi * divv)
