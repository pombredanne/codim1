from codim1.core.mesh import Mesh
from codim1.core.quad_strategy import QuadStrategy

def test_build_quadrature_list():
    msh = Mesh.simple_line_mesh(2)
    qs = QuadStrategy(msh, 2, 2, 2)

    assert(qs.quad_nonsingular.N == 2)

    assert(len(qs.quad_logr) == 2)
    assert(len(qs.quad_oneoverr) == 2)

    assert(qs.quad_logr[0].N == 2)
    assert(qs.quad_oneoverr[0].N == 2)

    assert(qs.quad_logr[0].x0 == qs.quad_nonsingular.x[0])
    assert(qs.quad_logr[1].x0 == qs.quad_nonsingular.x[1])
    assert(qs.quad_oneoverr[0].x0 == qs.quad_nonsingular.x[0])
    assert(qs.quad_oneoverr[1].x0 == qs.quad_nonsingular.x[1])

def test_get_quadrature_nonsingular():
    msh = Mesh.simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2)
    (Gqo, Gqi), (Hqo, Hqi) = qs.get_quadrature(0, 2)
    assert(Gqi[0] == qs.quad_nonsingular)
    assert(Gqo == qs.quad_nonsingular)
    assert(Hqi[0] == qs.quad_nonsingular)
    assert(Hqo == qs.quad_nonsingular)

def test_get_quadrature_singular():
    msh = Mesh.simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2)
    (Gqo, Gqi), (Hqo, Hqi) = qs.get_quadrature(0, 0)
    assert(Gqo == qs.quad_nonsingular)
    assert(Gqi == qs.quad_logr)
    assert(Hqo == qs.quad_nonsingular)
    assert(Hqi == qs.quad_oneoverr)

def test_get_quadrature_side():
    msh = Mesh.simple_line_mesh(3)
    qs = QuadStrategy(msh, 2, 2, 2)
    (Gqo, Gqi), (Hqo, Hqi) = qs.get_quadrature(1, 0)
    assert(Gqo == qs.quad_nonsingular)
    assert(Gqi[0] == qs.quad_shared_edge_right)
    assert(Hqo == qs.quad_nonsingular)
    assert(Hqi[0] == qs.quad_shared_edge_right)
