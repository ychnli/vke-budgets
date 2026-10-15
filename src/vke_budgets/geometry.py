"""
The shape of the Earth, and how horizontal derivatives use it.

A step (dlam, dphi) in longitude and latitude covers

    dx = N(phi) cos(phi) dlam    eastward
    dy = M(phi) dphi             northward

where M is the meridional (north-south) radius of curvature and N cos(phi) is
the radius of the latitude circle. On a sphere M = N = R. On an ellipsoid of
equatorial radius a and eccentricity e (e^2 = f(2 - f) for flattening f),

    M = a (1 - e^2) / (1 - e^2 sin^2 phi)^(3/2),    N = a / (1 - e^2 sin^2 phi)^(1/2).

Atmospheric models, MERRA-2's GEOS-5 included, solve their equations on a
sphere, so the default is a sphere with the model's radius. The WGS84
ellipsoid is here because the reference used it (through metpy).

`HorizontalGeometry` collects every choice the horizontal operators make. Its
defaults are the correct calculation; `REFERENCE` reproduces the reference
implementation ($REF/py/tools_comp.py, metpy 1.6.3), whose three flaws are
documented and measured in tests/test_reference_operator_flaws.py.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Earth:
    """An ellipsoid of revolution; flattening = 0 is a sphere.

    Attributes:
        a: equatorial radius (m).
        flattening: (a - b) / a, where b is the polar radius.
    """
    a: float
    flattening: float = 0.0

    @property
    def e2(self):
        """Eccentricity squared."""
        return self.flattening * (2.0 - self.flattening)

    def meridional_radius(self, lat_deg):
        """M(phi) (m): northward distance per radian of latitude."""
        s2 = np.sin(np.deg2rad(lat_deg)) ** 2
        return self.a * (1.0 - self.e2) / (1.0 - self.e2 * s2) ** 1.5

    def prime_vertical_radius(self, lat_deg):
        """N(phi) (m): N cos(phi) is the radius of the latitude circle."""
        s2 = np.sin(np.deg2rad(lat_deg)) ** 2
        return self.a / np.sqrt(1.0 - self.e2 * s2)

    def meridian_distance(self, lat_deg):
        """S(phi) (m), the distance north of the equator along a meridian,
        the integral of M dphi. Helmert's series in the third flattening
        n = f / (2 - f), accurate to well under a millimetre; on a sphere it
        is exactly a * phi."""
        phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        n = self.flattening / (2.0 - self.flattening)
        return self.a / (1.0 + n) * (
            (1.0 + n**2 / 4.0 + n**4 / 64.0) * phi
            - 1.5 * (n - n**3 / 8.0) * np.sin(2.0 * phi)
            + 15.0 / 16.0 * (n**2 - n**4 / 4.0) * np.sin(4.0 * phi)
            - 35.0 / 48.0 * n**3 * np.sin(6.0 * phi)
            + 315.0 / 512.0 * n**4 * np.sin(8.0 * phi))


# GEOS, the model behind MERRA-2, uses a sphere of radius MAPL_RADIUS =
# 6371.0e3 m (MAPL v2.0.0 of 2019 and today; the 2014 GEOS-5.12.4 that
# produced MERRA-2 has not been checked). The original NCL also assumes a
# sphere: its spherical harmonics and grad_latlon_cfd use 6.37122e6 m, a
# difference of 3e-5. The WGS84 ellipsoid came in through metpy.
MERRA2_SPHERE = Earth(a=6371.0e3)
WGS84 = Earth(a=6378137.0, flattening=1.0 / 298.257223563)
# metpy.constants.earth_avg_radius, which the reference uses in its
# divergence metric term.
METPY_MEAN_RADIUS = 6371008.7714


@dataclass(frozen=True)
class HorizontalGeometry:
    """Choices made by the horizontal derivative operators.

    Attributes:
        earth: the Earth model.
        periodic_lon: wrap the longitude derivative around the globe. None
            (the default) wraps when the grid is evenly spaced and covers the
            full 360 degrees, and uses one-sided differences at the edges
            otherwise. False reproduces reference flaw 3.
        metpy_dy_factor: multiply d/dy by a / M(phi) on top of the true
            spacing, as metpy 1.6.3 does without a CRS. Reference flaw 1:
            meridional derivatives come out 0.67% too large at the equator.
        metric_radius: radius (m) in the divergence metric term
            -v tan(phi) / r. None uses the Earth's own N(phi), which is
            correct. METPY_MEAN_RADIUS reproduces reference flaw 2.
    """
    earth: Earth = MERRA2_SPHERE
    periodic_lon: bool | None = None
    metpy_dy_factor: bool = False
    metric_radius: float | None = None


DEFAULT = HorizontalGeometry()
# Everything the reference does, flaws included.
REFERENCE = HorizontalGeometry(earth=WGS84, periodic_lon=False,
                               metpy_dy_factor=True, metric_radius=METPY_MEAN_RADIUS)
