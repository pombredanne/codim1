import numpy as np
from codim1.core import *

def test_vertex_connectivity():
    a = Vertex(np.ones(2))
    a.connect_to_element(1)
    assert(a.connected_to[0] == 1)
    a.connect_to_element(2)
    assert(a.connected_to[1] == 2)

def test_element_connectivity():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    c = Vertex(np.array([0.0, 1.0]))
    e = Element(a, b)
    e2 = Element(b, c)
    assert(a.connected_to[0] == e)
    assert(b.connected_to[0] == e)
    assert(b.connected_to[1] == e2)
    assert(c.connected_to[0] == e2)

def test_neighbors():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    c = Vertex(np.array([0.0, 1.0]))
    e = Element(a, b)
    e2 = Element(b, c)
    e.update_neighbors()
    e2.update_neighbors()
    assert(e.neighbors_left == [])
    assert(e.neighbors_right == [e2])
    assert(e2.neighbors_left == [e])
    assert(e2.neighbors_right == [])
