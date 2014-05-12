from codim1.core import *

def test_build_quadrature_list():
    msh = simple_line_mesh(2)
    qs = QuadStrategy(msh, 2, 2, 2, 2)

    assert(qs.quad_nonsingular[2].N == 2)

    assert(len(qs.quad_logr) == 2)
    assert(len(qs.quad_oneoverr) == 2)

    assert(qs.quad_logr[0].N == 2)
    assert(qs.quad_oneoverr[0].N == 2)

    assert(qs.quad_logr[0].x0 == qs.quad_nonsingular[2].x[0])
    assert(qs.quad_logr[1].x0 == qs.quad_nonsingular[2].x[1])
    assert(qs.quad_oneoverr[0].x0 == qs.quad_nonsingular[2].x[0])
    assert(qs.quad_oneoverr[1].x0 == qs.quad_nonsingular[2].x[1])

def test_get_quadrature_nonsingular():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', 0, 2)
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', 0, 2)
    assert(Gqi[0] == qs.quad_nonsingular[2])
    assert(Gqo == qs.quad_nonsingular[2])
    assert(Hqi[0] == qs.quad_nonsingular[2])
    assert(Hqo == qs.quad_nonsingular[2])

def test_get_quadrature_singular():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', 0, 0)
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', 0, 0)
    assert(Gqo == qs.quad_nonsingular[2])
    assert(Gqi == qs.quad_logr)
    assert(Hqo == qs.quad_nonsingular[2])
    assert(Hqi == qs.quad_oneoverr)

def test_get_quadrature_side():
    msh = simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2, 2)
    (Gqo, Gqi) = qs.get_quadrature('logr', 1, 0)
    (Hqo, Hqi) = qs.get_quadrature('oneoverr', 1, 0)
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

