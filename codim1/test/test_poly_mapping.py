import numpy as np
from codim1.core import *

def sine_boundary_func(v1, v2, t):
    x = (1 - t) * v1.loc[0] + t * v2.loc[0]
    x_length = v2.loc[0] - v1.loc[0]
    y = np.sin(np.pi * x / x_length)
    return np.array((x, y))

def test_quadratic_no_fnc_fail():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    try:
        lm = PolynomialMapping(m.elements[0], 2)
        assert(False)
    except:
        pass

def test_coeffs():
    m = simple_line_mesh(4)
    lm = PolynomialMapping(m.elements[2], 2, sine_boundary_func)
    assert(lm.coefficients[0, 1] == 0.25)
    assert(lm.coefficients[1, 1] == 1.0)

def test_get_phys_pts():
    m = simple_line_mesh(4)
    lm = PolynomialMapping(m.elements[2], 2, sine_boundary_func)

    # Element 2 should lie from 0 to 0.5
    pts = lm.get_physical_point(0.0)
    np.testing.assert_almost_equal(pts[0], 0.0)
    np.testing.assert_almost_equal(pts[1], 0.0)

    pts = lm.get_physical_point(0.5)
    np.testing.assert_almost_equal(pts[0], 0.25)
    np.testing.assert_almost_equal(pts[1], 1.0)

    pts = lm.get_physical_point(1.0)
    np.testing.assert_almost_equal(pts[0], 0.5)
    np.testing.assert_almost_equal(pts[1], 0.0)

def test_in_element_quadratic():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    lm = PolynomialMapping(m.elements[1], 2, sine_boundary_func)
    assert(lm.in_element((-0.5, 1.0))[0])
    assert(not lm.in_element((0.5, 0.0))[0])
    assert(not lm.in_element((0.5, 0.5))[0])

def test_in_element_corner():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    lm = PolynomialMapping(m.elements[1], 2, sine_boundary_func)
    assert(lm.in_element((1.0, -1.0))[0])
    assert(not lm.in_element((1.01, -1.0))[0])

def test_higher_mesh_not_linear():
    bf = BasisFunctions.from_degree(2)
    m = circular_mesh(2, 1.0, bf)
    assert(not m.is_linear)

def test_higher_order_coeff_gen():
    quad_map_gen = partial(PolynomialMapping,
                           degree = 2
                           boundary_function = )
    m = circular_mesh(2, 1.0, bf)
    coeffs_exact = np.array([[1.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]])
    np.testing.assert_almost_equal(m.coefficients[:, 0, :], coeffs_exact)

def test_higher_order_phys_pt():
    bf = BasisFunctions.from_degree(2)
    m = circular_mesh(2, 1.0, bf)
    phys_pt = m.get_physical_point(0, 0.5)
    np.testing.assert_almost_equal(phys_pt, (0.0, 1.0))
    phys_pt = m.get_physical_point(0, 0.25)
    np.testing.assert_almost_equal(phys_pt, (0.5, 0.75))
    phys_pt = m.get_physical_point(0, 0.75)
    np.testing.assert_almost_equal(phys_pt, (-0.5, 0.75))

def test_higher_order_jacobian():
    bf = BasisFunctions.from_degree(2)
    m = circular_mesh(2, 1.0, bf)
    x_hat = np.linspace(0, 1, 100)
    jacobian = m.get_jacobian(0, 0.5)
    np.testing.assert_almost_equal(jacobian, 2.0)

def test_higher_order_normal():
    bf = BasisFunctions.from_degree(2)
    m = circular_mesh(2, 1.0, bf)
    x_hat = np.linspace(0, 1, 100)
    normal = m.get_normal(0, 0.5)
    np.testing.assert_almost_equal(normal, (0.0, -1.0))
