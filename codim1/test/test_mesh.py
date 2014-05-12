import numpy as np
import warnings

from codim1.core.mesh import Mesh
from codim1.core.basis_funcs import BasisFunctions
from codim1.core.segment_distance import segments_distance
from codim1.core.mesh_gen import combine_meshes, simple_line_mesh,\
                                 circular_mesh

def test_get_neighbors():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    assert(m.get_neighbors(0, 'right') == m.element_objs[0].neighbors_right)

def test_in_element_doesnt_break_warnings():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    m.in_element(0, (-0.5, 0.0))
    with warnings.catch_warnings(record=True) as w:
        a = np.ones(2) / np.zeros(2)
    assert(len(w) == 1)

def test_in_element():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    assert(m.in_element(0, (-0.5, 0.0))[0])
    assert(not m.in_element(0, (0.5, 0.0))[0])
    assert(not m.in_element(0, (0.5, 0.5))[0])

def test_in_element_diag():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    assert(m.in_element(0, (-0.5, 0.5))[0])
    assert(not m.in_element(0, (0.5, 0.0))[0])
    assert(not m.in_element(0, (0.5, 0.5))[0])

def test_in_element_corner():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    assert(m.in_element(1, (1.0, -1.0))[0])
    assert(not m.in_element(1, (1.01, -1.0))[0])


def test_equivalent_pairs():
    m1 = simple_line_mesh(1, (-1.0, 0.0), (0.0, 1.0))
    m2 = simple_line_mesh(1, (0.0, 1.0), (1.0, 0.0))
    m = combine_meshes(m1, m2)
    equivalent_pairs = m._find_equivalent_pairs(1e-6)
    assert(equivalent_pairs[0, 0] == 1)
    assert(equivalent_pairs[0, 1] == 2)

def test_multisegment_mesh():
    m1 = simple_line_mesh(1, (-1.0, 0.0), (0.0, 1.0))
    m2 = simple_line_mesh(1, (0.0, 1.0), (1.0, 0.0))
    m = combine_meshes(m1, m2, ensure_continuity = True)
    assert(m.element_to_vertex[0, 0] == 0)
    assert(m.element_to_vertex[0, 1] == 1)
    assert(m.element_to_vertex[1, 0] == 1)
    assert(m.element_to_vertex[1, 1] == 3)
    assert(m.is_neighbor(0, 1, 'right'))
    assert(m.is_neighbor(1, 0, 'left'))

def test_get_phys_pts():
    m = simple_line_mesh(4)

    # Element 2 should lie from 0 to 0.5
    pts = m.get_physical_point(2, 0.5)
    np.testing.assert_almost_equal(pts[0], 0.25)
    pts = m.get_physical_point(2, 0.0)
    np.testing.assert_almost_equal(pts[0], 0.0)
    pts = m.get_physical_point(2, 1.0)
    np.testing.assert_almost_equal(pts[0], 0.5)
    np.testing.assert_almost_equal(pts[1], 0.0)

def test_jacobian():
    m = simple_line_mesh(4)
    j = m.get_jacobian(1, 0.0)
    np.testing.assert_almost_equal(j, 0.5)

def test_normals():
    m = simple_line_mesh(4)
    for i in range(4):
        assert(m.get_normal(0, 0.5)[0] == 0)
        assert(m.get_normal(0, 0.5)[1] == 1)

def test_connectivity():
    m = simple_line_mesh(4)
    assert(m.get_neighbors(0, 'left') == [])
    assert(m.get_neighbors(3, 'right') == [])

    assert(m.is_neighbor(2, 1, 'left'))
    assert(m.is_neighbor(2, 3, 'right'))

def test_connectivity_loop():
    vertices = np.array([(0.0, 1.0), (1.0, 0.0)])
    element_to_vertex = np.array([[0, 1], [1, 0]])
    m = Mesh(vertices, element_to_vertex)

    assert(m.is_neighbor(0, 1, 'left'))
    assert(m.is_neighbor(0, 1, 'right'))
    assert(m.is_neighbor(1, 0, 'left'))
    assert(m.is_neighbor(1, 0, 'right'))

def test_is_neighbor():
    m = simple_line_mesh(4)
    assert(m.is_neighbor(2, 3, 'right') == True)
    assert(m.is_neighbor(2, 3, 'left') == False)
    assert(m.is_neighbor(1, 0, 'left') == True)

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
    assert(m.element_distances[0, 3] == 1.0)
    assert(m.element_distances[0, 2] == 0.5)
    assert(m.element_distances[0, 1] == 0.0)
    assert(m.element_distances[0, 0] == 0.0)
    assert(m.element_distances[2, 0] == 0.5)
    np.testing.assert_almost_equal(
            m.element_distances.T - m.element_distances,
            np.zeros_like(m.element_distances))

def test_element_widths():
    vertices = np.array([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)])
    etov = np.array([(0, 1), (1, 2)])
    m = Mesh(vertices, etov)
    assert(m.element_widths[0] == 1.0)
    assert(m.element_widths[1] == 2.0)

def test_combine_meshes():
    m = simple_line_mesh(1)
    m2 = simple_line_mesh(1, (-2.0, 1.0), (0.0, 1.0))
    m3 = combine_meshes(m, m2)
    assert((m3.get_physical_point(0, 0.5) == (0.0, 0.0)).all())
    assert((m3.vertices[0] == (-1.0, 0.0)).all())
    assert((m3.vertices[1] == (1.0, 0.0)).all())
    assert((m3.vertices[2] == (-2.0, 1.0)).all())
    assert((m3.vertices[3] == (0.0, 1.0)).all())
    assert((m3.element_to_vertex[1] == (2, 3)).all())

def test_combine_mesh_parts():
    m = simple_line_mesh(1)
    m2 = simple_line_mesh(1, (-2.0, 1.0), (0.0, 1.0))
    m3 = combine_meshes(m, m2)
    assert(m3.parts[0] == m)
    assert(m3.parts[1] == m2)

def test_jacobian_type():
    m = simple_line_mesh(1)
    jacobian = m.get_jacobian(0, 0.0)
    assert(type(jacobian) == float)
