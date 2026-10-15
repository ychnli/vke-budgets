"""
Exact geometry of the WGS84 ellipsoid, and analytic test fields whose
horizontal derivatives are known in closed form.

On an ellipsoid of revolution (equatorial radius a, eccentricity e) two radii
of curvature set the distance per radian:

    M(phi) = a (1 - e^2) / (1 - e^2 sin^2 phi)^(3/2)    north-south (meridian)
    N(phi) = a / (1 - e^2 sin^2 phi)^(1/2)              east-west (prime vertical)

so a step (dlam, dphi) covers dx = N cos(phi) dlam eastward and dy = M dphi
northward, and

    df/dx = 1/(N cos phi) df/dlam,      df/dy = 1/M df/dphi.

The divergence of a horizontal vector (u, v) is

    div = 1/(M N cos phi) [ d(u M)/dlam + d(v N cos phi)/dphi ]
        = du/dx + dv/dy - v tan(phi) / N,

using d(N cos phi)/dphi = -M sin phi. On a sphere M = N = R and these reduce to
the familiar spherical formulas.

Nothing here uses metpy or pyproj, so the tests built on it are independent of
the code they check.
"""
import numpy as np
from scipy.integrate import quad

# WGS84, the ellipsoid metpy uses for '+proj=latlon'.
A_WGS84 = 6378137.0                      # m, equatorial radius
F_WGS84 = 1.0 / 298.257223563            # flattening
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)     # eccentricity squared

# metpy.constants.earth_avg_radius, the sphere the reference uses for its
# divergence metric term.
R_MEAN = 6371008.7714                    # m

# The MERRA-2 0.5 x 0.625 degree grid.
LAT_MERRA2 = np.linspace(-90.0, 90.0, 361)
LON_MERRA2 = -180.0 + 0.625 * np.arange(576)


def meridional_radius(lat_deg):
    """M(phi), the north-south radius of curvature (m)."""
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    return A_WGS84 * (1.0 - E2_WGS84) / (1.0 - E2_WGS84 * s2) ** 1.5


def prime_vertical_radius(lat_deg):
    """N(phi), the east-west radius of curvature (m). N cos(phi) is the
    radius of the latitude circle."""
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    return A_WGS84 / np.sqrt(1.0 - E2_WGS84 * s2)


def meridian_distance(lat_deg):
    """Distance north from the equator along a meridian (m):
    S(phi) = integral of M dphi' from 0 to phi. By construction dS/dy = 1."""
    lat_deg = np.atleast_1d(lat_deg)
    return np.array([quad(lambda p: meridional_radius(np.rad2deg(p)), 0.0, np.deg2rad(x),
                          epsabs=1e-9, epsrel=1e-13)[0]
                     for x in lat_deg])
