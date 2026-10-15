"""Earth radii and meridian distance against the independent formulas in
exact.py (the WGS84 meridian distance there is a numerical integral)."""
import numpy as np

import exact
from vke_budgets.geometry import MERRA2_SPHERE, WGS84

LAT = exact.LAT_MERRA2


def test_wgs84_radii():
    np.testing.assert_allclose(WGS84.meridional_radius(LAT), exact.meridional_radius(LAT),
                               rtol=1e-14)
    np.testing.assert_allclose(WGS84.prime_vertical_radius(LAT),
                               exact.prime_vertical_radius(LAT), rtol=1e-14)


def test_wgs84_meridian_distance_is_integral_of_m():
    # Helmert's series against quadrature, to 0.1 mm.
    np.testing.assert_allclose(WGS84.meridian_distance(LAT), exact.meridian_distance(LAT),
                               rtol=0, atol=1e-4)


def test_sphere():
    r = MERRA2_SPHERE.a
    np.testing.assert_allclose(MERRA2_SPHERE.meridional_radius(LAT), r, rtol=1e-15)
    np.testing.assert_allclose(MERRA2_SPHERE.prime_vertical_radius(LAT), r, rtol=1e-15)
    np.testing.assert_allclose(MERRA2_SPHERE.meridian_distance(LAT), r * np.deg2rad(LAT),
                               rtol=1e-15)
