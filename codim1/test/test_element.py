import numpy as np
from codim1.core import *

def test_apply_non_gen():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    e = Element(a, b)
    bf = lambda e: BasisFunctions.from_degree(2)
    apply_to_elements([e], "basis", bf)
    assert(e.basis == bf)

def test_apply_non_gen():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    e = Element(a, b)
    bf = BasisFunctions.from_degree(2),
    apply_to_elements([e], "basis", bf, non_gen = True)
    assert(e.basis == bf)

def test_element_init():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    e = Element(a, b)
    assert(type(e.mapping) == str)
    assert(type(e.basis) == str)
    assert(type(e.bc) == str)
    assert(type(e.continuous) == str)

def test_element_id():
    a = Vertex(np.ones(2))
    b = Vertex(np.array([1.0, 0.0]))
    e = Element(a, b)
    e.set_id(0)
    assert(e.id == 0)

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

def test_duplicate_connected_to():
    a = Vertex(np.ones(2))
    a.connect_to_element(1)
    a.connect_to_element(1)
    assert(a.connected_to == [1])

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

def test_vertex_param():
    a = Vertex(np.ones(2), 1.0)
    assert(a.param == 1.0)


def test_vertex_loc_type():
    v = Vertex((1.0, 2.0))
    assert(type(v.loc) == np.ndarray)

def test_vertex_connectivity():
    a = Vertex(np.ones(2))
    a.connect_to_element(1)
    assert(a.connected_to[0] == 1)
    a.connect_to_element(2)
    assert(a.connected_to[1] == 2)

