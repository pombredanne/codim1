from functools import partial
import numpy as np
from codim1.core import *


def test_quadratic_no_fnc_fail():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    try:
        lm = PolynomialMapping(m.elements[0], 2)
        assert(False)
    except:
        pass

def test_coeffs():
    m = simple_line_mesh(4)
    sine_boundary_func = lambda x: (x, np.sin(x))
    lm = PolynomialMapping(m.elements[2], 2, sine_boundary_func)
    assert(lm.coefficients[0, 1] == 0.25)
    assert(lm.coefficients[1, 1] == np.sin(0.25))

def test_get_phys_pts():
    m = simple_line_mesh(4)
    sine_boundary_func = lambda x: (x, np.sin(x))
    lm = PolynomialMapping(m.elements[2], 2, sine_boundary_func)

    # Element 2 should lie from 0 to 0.5
    pts = lm.get_physical_point(0.0)
    np.testing.assert_almost_equal(pts[0], 0.0)
    np.testing.assert_almost_equal(pts[1], 0.0)

    pts = lm.get_physical_point(0.5)
    np.testing.assert_almost_equal(pts[0], 0.25)
    np.testing.assert_almost_equal(pts[1], np.sin(0.25))

    pts = lm.get_physical_point(1.0)
    np.testing.assert_almost_equal(pts[0], 0.5)
    np.testing.assert_almost_equal(pts[1], 0.0)

def test_higher_order_coeff_gen():
    quad_map_gen = partial(PolynomialMapping, degree = 2)
    m = circular_mesh(2, 1.0, quad_map_gen)
    coeffs_exact = np.array([[1.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]])
    np.testing.assert_almost_equal(m.elements[0].mapping.coefficients,
                                   coeffs_exact)

def test_higher_order_phys_pt():
    quad_map_gen = partial(PolynomialMapping, degree = 2)
    m = circular_mesh(2, 1.0, quad_map_gen)
    phys_pt = m.elements[0].mapping.get_physical_point(0.5)
    np.testing.assert_almost_equal(phys_pt, (0.0, 1.0))
    phys_pt = m.elements[0].mapping.get_physical_point(0.25)
    np.testing.assert_almost_equal(phys_pt, (0.5, 0.75))
    phys_pt = m.elements[0].mapping.get_physical_point(0.75)
    np.testing.assert_almost_equal(phys_pt, (-0.5, 0.75))

def test_higher_order_jacobian():
    quad_map_gen = partial(PolynomialMapping, degree = 2)
    m = circular_mesh(2, 1.0, quad_map_gen)
    x_hat = np.linspace(0, 1, 100)
    jacobian = m.elements[0].mapping.get_jacobian(0.5)
    np.testing.assert_almost_equal(jacobian, 2.0)

def test_higher_order_normal():
    quad_map_gen = partial(PolynomialMapping, degree = 2)
    m = circular_mesh(2, 1.0, quad_map_gen)
    x_hat = np.linspace(0, 1, 100)
    normal = m.elements[0].mapping.get_normal(0.5)
    np.testing.assert_almost_equal(normal, (0.0, -1.0))
