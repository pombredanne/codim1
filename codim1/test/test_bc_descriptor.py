from codim1.core import *

def test_constant_bc():
    cbc = ConstantBC("traction", 2.0, 1)
    assert(cbc.get_value(0) == 2.0)
    assert(cbc.element == 1)

def test_zero_bc():
    cbc = ZeroBC("traction", 1)
    assert(cbc.get_value(0) == 0.0)
    assert(cbc.get_value(0.5) == 0.0)
    assert(cbc.get_value(0.9) == 0.0)
    assert(cbc.element == 1)

def test_bc_descriptor_all():
    simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))


