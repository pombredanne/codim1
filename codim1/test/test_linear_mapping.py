import numpy as np
import warnings
from codim1.core import *

def test_in_element_doesnt_break_warnings():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    lm = PolynomialMapping(m.elements[0])
    lm.in_element((-0.5, 0.0))
    with warnings.catch_warnings(record=True) as w:
        a = np.ones(2) / np.zeros(2)
    assert(len(w) == 1)

def test_in_element():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    lm = PolynomialMapping(m.elements[0])
    assert(lm.in_element((-0.5, 0.0))[0])
    assert(not lm.in_element((0.5, 0.0))[0])
    assert(not lm.in_element((0.5, 0.5))[0])

def test_in_element_diag():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    lm = PolynomialMapping(m.elements[0])
    assert(lm.in_element((-0.5, 0.5))[0])
    assert(not lm.in_element((0.5, 0.0))[0])
    assert(not lm.in_element((0.5, 0.5))[0])

def test_in_element_corner():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    lm = PolynomialMapping(m.elements[1])
    assert(lm.in_element((1.0, -1.0))[0])
    assert(not lm.in_element((1.01, -1.0))[0])

def test_jacobian_type():
    m = simple_line_mesh(1)
    lm = PolynomialMapping(m.elements[0])
    jacobian = lm.get_jacobian(0.0)
    assert(type(jacobian) == float)

def test_get_phys_pts():
    m = simple_line_mesh(4)
    lm = PolynomialMapping(m.elements[2])

    # Element 2 should lie from 0 to 0.5
    pts = lm.get_physical_point(0.5)
    np.testing.assert_almost_equal(pts[0], 0.25)
    pts = lm.get_physical_point(0.0)
    np.testing.assert_almost_equal(pts[0], 0.0)
    pts = lm.get_physical_point(1.0)
    np.testing.assert_almost_equal(pts[0], 0.5)
    np.testing.assert_almost_equal(pts[1], 0.0)

def test_jacobian():
    m = simple_line_mesh(4)
    lm = PolynomialMapping(m.elements[1])
    j = lm.get_jacobian(0.0)
    np.testing.assert_almost_equal(j, 0.5)

def test_normals():
    m = simple_line_mesh(4)
    for i in range(4):
        lm = PolynomialMapping(m.elements[i])
        assert(lm.get_normal(0.5)[0] == 0)
        assert(lm.get_normal(0.5)[1] == 1)

def test_apply_mapping():
    m = simple_line_mesh(4)
    apply_mapping(m, PolynomialMapping)
    for i in range(4):
        assert(m.elements[i].mapping.get_normal(0.5)[0] == 0)
        assert(m.elements[i].mapping.get_normal(0.5)[1] == 1)

def test_get_linear_approx():
    m = simple_line_mesh(4)
    apply_mapping(m, PolynomialMapping)
    for i in range(4):
        verts = m.elements[i].mapping.get_linear_approximation()
        np.testing.assert_almost_equal(verts[0].loc,
                                       m.elements[i].vertex1.loc)
        np.testing.assert_almost_equal(verts[1].loc,
                                       m.elements[i].vertex2.loc)

def test_distance_between_mappings():
    m = simple_line_mesh(4)
    d = distance_between_mappings(m.elements[0].mapping,
                                  m.elements[3].mapping)
    np.testing.assert_almost_equal(d, 1.0)
