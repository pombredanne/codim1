import numpy as np
from codim1.core import *
from codim1.core.segment_distance import segments_distance
from codim1.fast_lib import aligned_single_integral, single_integral,\
                    TractionKernel, ConstantBasis

def test_segment_distance():
    v1 = (0, 0)
    v2 = (0, 1)
    v3 = (1, 0)
    v4 = (1, 1)
    dist = segments_distance(v1[0], v1[1], v2[0], v2[1],
                             v3[0], v3[1], v4[0], v4[1])
    assert(dist == 1.0)
    dist2 = segments_distance(v3[0], v3[1], v4[0], v4[1],
                             v1[0], v1[1], v2[0], v2[1])
    assert(dist2 == 1.0)

def test_element_distances():
    m = simple_line_mesh(4)
    qs = QuadStrategy(m, 2, 2, 2, 2)
    assert(qs.element_distances[0, 3] == 1.0)
    assert(qs.element_distances[0, 2] == 0.5)
    assert(qs.element_distances[0, 1] == 0.0)
    assert(qs.element_distances[0, 0] == 0.0)
    assert(qs.element_distances[2, 0] == 0.5)
    np.testing.assert_almost_equal(
            qs.element_distances.T - qs.element_distances,
            np.zeros_like(qs.element_distances))

def test_build_quadrature_list():
    msh = simple_line_mesh(2)
    qs = QuadStrategy(msh, 2, 2, 2, 2)

    assert(len(qs.quad_nonsingular[2].x) == 2)

    assert(len(qs.quad_logr) == 2)
    assert(len(qs.quad_oneoverr) == 2)

    assert(len(qs.quad_logr[0].x) == 2)
    assert(len(qs.quad_oneoverr[0].x) == 6)

    assert(qs.quad_logr[0].x0 == qs.quad_nonsingular[2].x[0])
    assert(qs.quad_logr[1].x0 == qs.quad_nonsingular[2].x[1])
    assert(qs.quad_oneoverr[0].x0 == qs.quad_nonsingular[2].x[0])
    assert(qs.quad_oneoverr[1].x0 == qs.quad_nonsingular[2].x[1])

def test_get_quadrature_nonsingular():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', msh.elements[0],
                                           msh.elements[2])
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', msh.elements[0],
                                               msh.elements[2])
    assert(Gqi[0] == qs.quad_nonsingular[2])
    assert(Gqo == qs.quad_nonsingular[2])
    assert(Hqi[0] == qs.quad_nonsingular[2])
    assert(Hqo == qs.quad_nonsingular[2])

def test_get_quadrature_singular():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', msh.elements[0],
                                           msh.elements[0])
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', msh.elements[0],
                                               msh.elements[0])
    assert(Gqo == qs.quad_nonsingular[2])
    assert(Gqi == qs.quad_logr)
    assert(Hqo == qs.quad_nonsingular[2])
    assert(Hqi == qs.quad_oneoverr)

def test_get_quadrature_side():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', msh.elements[1],
                                           msh.elements[0])
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', msh.elements[1],
                                               msh.elements[0])
    assert(Gqo == qs.quad_nonsingular[2])
    assert(Gqi[0] == qs.quad_shared_edge_right)
    assert(Hqo == qs.quad_nonsingular[2])
    assert(Hqi[0] == qs.quad_shared_edge_right)

def test_quad_strategy_nonsingular():
    msh = simple_line_mesh(4)
    qs = QuadStrategy(msh, 2, 5, 2, 2)
    assert(qs.choose_nonsingular(0, 0) == 5)
    assert(qs.choose_nonsingular(0, 1) == 5)
    assert(qs.choose_nonsingular(0, 2) == 4)
    assert(qs.choose_nonsingular(0, 3) == 3)

def test_quad_strat_min():
    msh = simple_line_mesh(20)
    qs = QuadStrategy(msh, 8, 10, 2, 2)
    assert(qs.choose_nonsingular(0, 15) == 8)

def test_gll_quad_strategy():
    msh = simple_line_mesh(2)
    qs = GLLQuadStrategy(msh, 8, 10, 2, 2)
    np.testing.assert_almost_equal(qs.get_simple().x[0], 0.0)
    np.testing.assert_almost_equal(qs.get_simple().x[-1], 1.0)

def test_gll_quad_strategy_interior():
    msh = simple_line_mesh(2)
    qs = GLLQuadStrategy(msh, 4, 10, 2, 2)
    q = qs.get_interior_quadrature(msh.elements[0], (-50.0, 0.0))
    assert(np.array(q.x).size == 4)

def test_telles_quad_strategy():
    msh = simple_line_mesh(4)
    tqs = TellesQuadStrategy(4)
    qi = tqs.get_interior_quadrature(msh.elements[3], [0.0, 0.0])

    kernel = TractionKernel(1.0, 0.25)
    one = ConstantBasis((1.0, 1.0))
    est = single_integral(msh.elements[3].mapping.eval,
                          kernel, one, one, qi, 0, 0)
    exact = -0.0367726
    np.testing.assert_almost_equal(exact, est[0][1])
