import numpy as np

from codim1.core.mesh import Mesh
from codim1.core.basis_funcs import BasisFunctions
from codim1.core.mesh_gen import combine_meshes, simple_line_mesh,\
                                 circular_mesh, from_vertices_and_etov

def test_get_neighbors():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    assert(m.get_neighbors(0, 'right') == m.elements[0].neighbors_right)

def test_equivalent_pairs():
    m1 = simple_line_mesh(1, (-1.0, 0.0), (0.0, 1.0))
    m2 = simple_line_mesh(1, (0.0, 1.0), (1.0, 0.0))
    m = combine_meshes(m1, m2)
    equivalent_pairs = m._find_equivalent_pairs(1e-6)
    assert(equivalent_pairs[0][0] == m.vertices[1])
    assert(equivalent_pairs[0][1] == m.vertices[2])

def test_multisegment_mesh():
    m1 = simple_line_mesh(1, (-1.0, 0.0), (0.0, 1.0))
    m2 = simple_line_mesh(1, (0.0, 1.0), (1.0, 0.0))
    m = combine_meshes(m1, m2, ensure_continuity = True)
    assert(m.elements[0].vertex1 == m.vertices[0])
    assert(m.elements[0].vertex2 == m.vertices[1])
    assert(m.elements[1].vertex1 == m.vertices[1])
    assert(m.elements[1].vertex2 == m.vertices[3])
    assert(m.is_neighbor(0, 1, 'right'))
    assert(m.is_neighbor(1, 0, 'left'))

def test_connectivity():
    m = simple_line_mesh(4)
    assert(m.get_neighbors(0, 'left') == [])
    assert(m.get_neighbors(3, 'right') == [])

    assert(m.is_neighbor(2, 1, 'left'))
    assert(m.is_neighbor(2, 3, 'right'))

def test_connectivity_loop():
    vertices = np.array([(0.0, 1.0), (1.0, 0.0)])
    element_to_vertex = np.array([[0, 1], [1, 0]])
    m = from_vertices_and_etov(vertices, element_to_vertex)

    assert(m.is_neighbor(0, 1, 'left'))
    assert(m.is_neighbor(0, 1, 'right'))
    assert(m.is_neighbor(1, 0, 'left'))
    assert(m.is_neighbor(1, 0, 'right'))

def test_is_neighbor():
    m = simple_line_mesh(4)
    assert(m.is_neighbor(2, 3, 'right') == True)
    assert(m.is_neighbor(2, 3, 'left') == False)
    assert(m.is_neighbor(1, 0, 'left') == True)


def test_element_widths():
    vertices = np.array([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)])
    etov = np.array([(0, 1), (1, 2)])
    m = from_vertices_and_etov(vertices, etov)
    assert(m.elements[0].length == 1.0)
    assert(m.elements[1].length == 2.0)

def test_combine_meshes_vertex_loc_type():
    m1 = simple_line_mesh(1)
    m2 = simple_line_mesh(1, (-2.0, 1.0), (0.0, 1.0))
    m3 = combine_meshes(m1, m2)
    assert(type(m1.vertices[0].loc) == np.ndarray)
    assert(type(m2.vertices[0].loc) == np.ndarray)
    assert(type(m3.vertices[0].loc) == np.ndarray)


def test_combine_meshes():
    m = simple_line_mesh(1)
    m2 = simple_line_mesh(1, (-2.0, 1.0), (0.0, 1.0))
    m3 = combine_meshes(m, m2)
    assert((m3.vertices[0].loc == (-1.0, 0.0)).all())
    assert((m3.vertices[1].loc == (1.0, 0.0)).all())
    assert((m3.vertices[2].loc == (-2.0, 1.0)).all())
    assert((m3.vertices[3].loc == (0.0, 1.0)).all())
    assert(m3.elements[1].vertex1 == m3.vertices[2])
    assert(m3.elements[1].vertex2 == m3.vertices[3])


