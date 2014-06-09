import pytest
import numpy as np
from functools import partial
from codim1.assembly.sgbem import _make_which_kernels, sgbem_assemble,\
    _element_mass_rhs
from codim1.assembly import simple_matrix_assemble, simple_rhs_assemble,\
    mass_matrix_for_rhs, assemble_mass_matrix
from codim1.core import *
from codim1.fast_lib import ConstantBasis

def test_which_kernels():
    ek = ElasticKernelSet(1.0, 0.25)
    wk = _make_which_kernels(ek)
    # Just a cursory test to make sure everything was assembled properly.
    assert(wk["displacement"]["traction"]["rhs"][0] == ek.k_d)

def make_mesh():
    mesh = simple_line_mesh(2)
    bf = basis_funcs.basis_from_degree(1)
    qs = QuadStrategy(mesh, 6, 6, 6, 6)
    apply_to_elements(mesh, "basis", bf, non_gen = True)
    apply_to_elements(mesh, "continuous", True, non_gen = True)
    apply_to_elements(mesh, "qs", qs, non_gen = True)
    bc_left = BC("displacement", ConstantBasis([1.0, 1.0]))
    bc_right = BC("traction", ConstantBasis([1.0, 1.0]))
    mesh.elements[0].bc = bc_left
    mesh.elements[1].bc = bc_right
    init_dofs(mesh)
    return mesh

def test_sgbem_mass_rhs():
    mesh = make_mesh()
    rhs_matrix = np.zeros((mesh.total_dofs, mesh.total_dofs))
    _element_mass_rhs(rhs_matrix, mesh.elements[0])
    _element_mass_rhs(rhs_matrix, mesh.elements[1])
    np.testing.assert_almost_equal(0.5 * np.sum(rhs_matrix, axis = 1),
                                   [0.25,0.5,0.25,0.25,0.5,0.25])

def test_sgbem_assemble():
    mesh = make_mesh()
    elastic_k = ElasticKernelSet(1.0, 0.25)
    matrix, rhs = sgbem_assemble(mesh, elastic_k)
