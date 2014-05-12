import numpy as np
from codim1.core import *

def test_circular_mesh():
    a = circular_mesh(4, 1.0)
    np.testing.assert_almost_equal(a.vertices[1].loc[0], 0.0)
    np.testing.assert_almost_equal(a.vertices[1].loc[1], 1.0)

def test_simple_line_mesh():
    m = simple_line_mesh(2)
    correct_vertices = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    correct_etov = np.array([[0, 1], [1, 2]])
    for i in range(correct_vertices.shape[0]):
        assert((m.vertices[i].loc == correct_vertices[i, :]).all())
        assert(m.elements.vertex1 == m.vertices[correct_etov[i, 0]])
        assert(m.elements.vertex2 == m.vertices[correct_etov[i, 1]])

def test_simple_line_mesh():
    m = simple_line_mesh(2, (3.0, 0.0), (4.0, 0.0))
    correct_vertices = np.array([[3.0, 0.0], [3.5, 0.0], [4.0, 0.0]])
    for i in range(correct_vertices.shape[0]):
        assert((m.vertices[i].loc == correct_vertices[i, :]).all())

def test_angular_simple_line_mesh():
    m = simple_line_mesh(2, (-1.0, 1.0), (1.0, -1.0))
    assert(m.vertices[0].loc[0] == -1.0)
    assert(m.vertices[0].loc[1] == 1.0)
    assert(m.vertices[1].loc[0] == 0.0)
    assert(m.vertices[1].loc[1] == 0.0)
    assert(m.vertices[2].loc[0] == 1.0)
    assert(m.vertices[2].loc[1] == -1.0)
