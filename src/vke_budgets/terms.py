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

Substituting the momentum equation,

    du/dt = -u . grad(u) - omega du/dp - grad(Phi) - f k x u + F,

and noting that u . (u . grad u) = u . grad K and u . (k x u) = 0 (the
Coriolis force does no work), the momentum side becomes

    HAKE = -w_m u . grad K                    horizontal advection of K
    VAKE = -w_m omega (u du/dp + v dv/dp)     vertical advection of K
    PEKE = -w_m u . grad Phi                  conversion of potential energy to K
    xOKE =  w_m u . F_x                       momentum physics, process x

Substituting the moisture equation dq/dt = -u . grad q - omega dq/dp + S,
the vapor side becomes

    HAV  = -w_v u . grad q                    horizontal advection of q
    VAV  = -w_v omega dq/dp                   vertical advection of q
    xOV  =  w_v S_x                           moisture physics, process x

Flux form. The convergence terms

    HCV  = -K div(q^n u),     VCV = -K d(q^n omega)/dp

are chosen so that HAKE + HCV = -div(E u) and VAKE + VCV = -d(E omega)/dp
exactly: hflux = HAKE + HCV and vflux = VAKE + VCV are the horizontal and
vertical flux divergences of E. Continuity, div(u) + d(omega)/dp = 0, makes
HCV + VCV = HAV + VAV; on real data the two pairs differ by the discrete
continuity error E (div u + d omega/dp).

For n = 2 the reference also writes HCV_vO = -w_v div(q u) and
VCV_vO = -w_v d(q omega)/dp, the form commented out in the original NCL.
Their sum differs from HCV + VCV by q^2 K (div u + d omega/dp), which
vanishes only by continuity.

The functions below take DataArrays on pressure levels and are pure.
"""
from .horizontal import divergence, gradient


def kinetic_energy(u, v):
    """K = (u^2 + v^2) / 2 (m2 s-2)."""
    return 0.5 * (u * u + v * v)


def weights(q, k, n):
    """(w_m, w_v) = (q^n, n q^(n-1) K): the weights of the momentum and
    vapor tendencies in dE/dt, for E = q^n K."""
    w_m = q**n
    w_v = k if n == 1 else n * q ** (n - 1) * k
    return w_m, w_v


# --- Momentum side --------------------------------------------------------

def horizontal_advection_of_ke(w_m, u, v, k, geometry):
    """HAKE = -w_m (u dK/dx + v dK/dy)."""
    dkdx, dkdy = gradient(k, geometry)
    return -w_m * (u * dkdx + v * dkdy)


def vertical_advection_of_ke(w_m, u, v, omega_dudp, omega_dvdp):
    """VAKE = -w_m (u omega du/dp + v omega dv/dp). The products omega du/dp
    are formed on model levels, where the vertical derivative is taken, and
    then interpolated (see budget.py)."""
    return -w_m * (u * omega_dudp + v * omega_dvdp)


def pe_to_ke_conversion(w_m, u, v, phi, geometry):
    """PEKE = -w_m (u dPhi/dx + v dPhi/dy): work done by the pressure-gradient
    force, which on pressure surfaces is the geopotential gradient. It
    converts potential energy into K."""
    dphidx, dphidy = gradient(phi, geometry)
    return -w_m * (u * dphidx + v * dphidy)


def momentum_physics(w_m, u, v, dudt, dvdt):
    """w_m (u du/dt + v dv/dt) for a physics tendency (du/dt, dv/dt) of the
    wind: MOKE (moist), TOKE (turbulence), GOKE (gravity-wave drag), DOKE
    (MERRA-2's dynamics), and the momentum half of AOKEV (analysis)."""
    return w_m * (u * dudt + v * dvdt)


# --- Vapor side -----------------------------------------------------------

def horizontal_advection_of_vapor(w_v, u, v, q, geometry):
    """HAV = -w_v (u dq/dx + v dq/dy)."""
    dqdx, dqdy = gradient(q, geometry)
    return -w_v * (u * dqdx + v * dqdy)


def vertical_advection_of_vapor(w_v, omega_dqdp):
    """VAV = -w_v omega dq/dp, with omega dq/dp formed on model levels."""
    return -w_v * omega_dqdp


def horizontal_convergence_of_vapor(k, w_m, u, v, geometry):
    """HCV = -K div(q^n u), where w_m = q^n. With HAKE it makes the
    horizontal flux divergence -div(E u)."""
    return -k * divergence(w_m * u, w_m * v, geometry)


def vertical_convergence_of_vapor(k, d_wm_omega_dp):
    """VCV = -K d(q^n omega)/dp, with the derivative taken on model levels.
    With VAKE it makes the vertical flux divergence -d(E omega)/dp."""
    return -k * d_wm_omega_dp


def ncl_convergence_of_vapor(w_v, q, u, v, d_q_omega_dp, geometry):
    """(HCV_vO, VCV_vO) = (-w_v div(q u), -w_v d(q omega)/dp): the form
    commented out in the original NCL, kept for comparison. For n = 1 it is
    the same as (HCV, VCV)."""
    return -w_v * divergence(q * u, q * v, geometry), -w_v * d_q_omega_dp


def moisture_physics(w_v, dqdt):
    """w_v dq/dt for a physics tendency of q: POV (moist), TOV (turbulence),
    COV (chemistry), DOV (MERRA-2's dynamics), and the vapor half of AOKEV."""
    return w_v * dqdt


# --- Tendencies -----------------------------------------------------------

def local_tendency(w_m, w_v, u, v, dqdt, dudt, dvdt):
    """dE/dt = w_m (u du/dt + v dv/dt) + w_v dq/dt, from the change of the
    instantaneous q, u, v across the window (KEVTEND, VTETEND)."""
    return w_m * (u * dudt + v * dvdt) + w_v * dqdt


def surface_pressure_term(dpsdt, e_bottom, g):
    """SPTE = (dps/dt) E(p_bottom) / g (kg s-3): the Leibniz term for a column
    whose bottom moves with the surface pressure,

        d/dt <E> = <dE/dt> + E(p_bottom) (dp_bottom/dt) / g.

    As in the reference, E(p_bottom) is E at the lowest level above ground
    and the term is applied everywhere."""
    return dpsdt * e_bottom / g


# --- Flux-form decomposition (KEV only; diagnostics) -----------------------
#
# The flux divergence of KEV = qK splits by the product rule as
#
#     div3(q K v) = q div3(K v) + K div3(q v) - q K div3(v),
#
# with div3(a v) = div(a u) + d(a omega)/dp. Its horizontal and vertical
# parts, with a minus sign, are
#
#     HFOKE + HFOV + HFADT = -div(q K u)          (HFOV is HCV)
#     VFOKE + VFOV + VFADT = -d(q K omega)/dp     (VFOV is VCV)
#
# and equal HAKE + HCV and VAKE + VCV analytically, but are differenced in a
# different order. DIVV = div(u) + d(omega)/dp vanishes by continuity, so
# every term proportional to it measures the discrete continuity error.

def horizontal_flux_of_ke(q, k, u, v, geometry):
    """HFOKE = -q div(K u)."""
    return -q * divergence(k * u, k * v, geometry)


def vertical_flux_of_ke(q, d_k_omega_dp):
    """VFOKE = -q d(K omega)/dp, the derivative taken on model levels."""
    return -q * d_k_omega_dp


def horizontal_divergence_cross_term(q, k, u, v, geometry):
    """HFADT = q K div(u)."""
    return q * k * divergence(u, v, geometry)


def vertical_divergence_cross_term(q, k, d_omega_dp):
    """VFADT = q K d(omega)/dp."""
    return q * k * d_omega_dp


def three_d_divergence(u, v, d_omega_dp, geometry):
    """DIVV = div(u) + d(omega)/dp (s-1), zero by continuity."""
    return divergence(u, v, geometry) + d_omega_dp


def geopotential_flux_terms(q, phi, u, v, omega, d_phi_dp, d_omega_phi_dp, divv, geometry):
    """The flux form of PEKE = -q u . grad(Phi). With the hydrostatic
    specific volume alpha = -dPhi/dp,

        PEKE = FOGEOP + BARCC + QPHIDIVV,
        HFOGEOP  = -q div(Phi u)
        FOGEOP   = HFOGEOP - q d(omega Phi)/dp     flux of geopotential
        BARCC    = -omega alpha q                   baroclinic conversion
        QPHIDIVV =  q Phi DIVV                      continuity closure

    Returns (ALPHA, BARCC, HFOGEOP, FOGEOP, QPHIDIVV)."""
    alpha = -d_phi_dp
    hfogeop = -q * divergence(phi * u, phi * v, geometry)
    return (alpha, -omega * alpha * q, hfogeop, hfogeop - q * d_omega_phi_dp,
            q * phi * divv)
