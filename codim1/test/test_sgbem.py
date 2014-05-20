from functools import partial
from codim1.assembly.sgbem import _make_which_kernels, sgbem_assemble
from codim1.core import *

def test_which_kernels():
    ek = ElasticKernelSet(1.0, 0.25)
    wk = _make_which_kernels(ek)
    # Just a cursory test to make sure everything was assembled properly.
    assert(wk["displacement"]["traction"]["rhs"][0] == ek.k_d)


def test_sgbem_assemble():
    mesh = simple_line_mesh(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    apply_to_elements(mesh, "basis", bf, non_gen = True)
    apply_to_elements(mesh, "continuous", True, non_gen = True)
    init_dofs(mesh)

    bc_left = partial(ConstantBC, "disp", 1.0)
    bc_right = partial(ConstantBC, "trac", 1.0)
    mesh.elements[0].bc = ConstantBC("disp", 1.0, mesh.elements[0])
    mesh.elements[1].bc = ConstantBC("trac", 2.0, mesh.elements[0])

    qs = QuadStrategy(mesh, 6, 6, 6, 6)
    ek = ElasticKernelSet(1.0, 0.25)
    matrix, rhs = sgbem_assemble(mesh, qs, ek)
