from codim1.core import *
from codim1.fast_lib import ConstantBasis, ZeroBasis

def test_constant_bc():
    basis = ConstantBasis([2.0, 2.0])
    cbc = BC("traction", basis)
    assert(cbc.basis == basis)

def test_bc_descriptor_all():
    m = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    cbc = BC("traction", ZeroBasis())
    apply_to_elements(m, "bc", cbc, non_gen = True)
    assert(m.elements[0].bc == cbc)
    assert(m.elements[1].bc == cbc)
