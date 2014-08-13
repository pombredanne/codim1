from codim1.core.new_mesh import *

def test_prototype():
    """
    Some confirmation code for the protoype of the new meshing structures.
    """
    pt2d = Vertex([1.0, 1.0])
    pt2d2 = Vertex([0.0, 1.0])
    pt3d = Vertex([1.0, 1.0, 2.0])

    e2d = Edge([pt2d, pt2d2])
    new_edges = e2d.refine()
    assert(new_edges[0].components[0].loc == [1.0,1.0])
    assert(new_edges[0].components[1].loc == [0.5,1.0])
    assert(new_edges[1].components[0].loc == [0.5,1.0])
    assert(new_edges[1].components[1].loc == [0.0,1.0])
