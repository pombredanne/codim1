import numpy as np
from codim1.core.mesh import Mesh

def test_simple_line_mesh():
    m = Mesh.simple_line_mesh(2)
    correct_vertices = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    correct_etov = np.array([[0, 1], [1, 2]])
    assert((m.vertices == correct_vertices).all())
    assert((m.element_to_vertex == correct_etov).all())
    assert(m.element_to_vertex.dtype.type is np.int64)

def test_simple_line_mesh():
    m = Mesh.simple_line_mesh(2, 3.0, 4.0)
    correct_vertices = np.array([[3.0, 0.0], [3.5, 0.0], [4.0, 0.0]])
    assert((m.vertices == correct_vertices).all())

def test_get_phys_pts():
    m = Mesh.simple_line_mesh(4)

    # Element 2 should lie from 0 to 0.5
    pts = m.get_physical_points(2, 0.5)
    np.testing.assert_almost_equal(pts[0], 0.25)
    pts = m.get_physical_points(2, 0.0)
    np.testing.assert_almost_equal(pts[0], 0.0)
    pts = m.get_physical_points(2, 1.0)
    np.testing.assert_almost_equal(pts[0], 0.5)
    np.testing.assert_almost_equal(pts[1], 0.0)

def test_get_one_phys_pts():
    m = Mesh.simple_line_mesh(4)
    # Element 2 should lie from 0 to 0.5
    pts = m.get_physical_points(2, np.array([0.5]))
    np.testing.assert_almost_equal(pts[0], 0.25)

def test_jacobian():
    m = Mesh.simple_line_mesh(4)
    j = m.get_element_jacobian(1)
    np.testing.assert_almost_equal(j, 0.5)

def test_normals():
    m = Mesh.simple_line_mesh(4)
    assert(m.normals.shape[0] == 4)
    assert((m.normals[:, 1] == [1, 1, 1, 1]).all())

def test_connectivity():
    m = Mesh.simple_line_mesh(4)
    assert(m.neighbors.dtype == np.int)
    assert(m.neighbors[0][0] == -1)
    assert(m.neighbors[3][1] == -1)

    assert(m.neighbors[2][0] == 1)
    assert(m.neighbors[2][1] == 3)

def test_connectivity_loop():
    vertices = np.array([[0, 1], [1, 0]])
    element_to_vertex = np.array([[0, 1], [1, 0]])
    m = Mesh(vertices, element_to_vertex)

    assert(m.neighbors.dtype == np.int)
    assert(m.neighbors[0][0] == 1)
    assert(m.neighbors[0][1] == 1)
    assert(m.neighbors[1][0] == 0)
    assert(m.neighbors[1][1] == 0)

def test_is_neighbor():
    m = Mesh.simple_line_mesh(4)
    assert(m.is_neighbor(2, 3, 'right') == True)
    assert(m.is_neighbor(2, 3, 'left') == False)
    assert(m.is_neighbor(1, 0, 'left') == True)
