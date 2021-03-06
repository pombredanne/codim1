import numpy as np
from codim1.assembly import mass_matrix_for_rhs, assemble_mass_matrix
from codim1.core import *
import codim1.core.basis_funcs as basis_funcs
import codim1.core.dof_handler as dof_handler
import codim1.core.quadrature as quadrature
from codim1.fast_lib import CoeffBasis
from codim1.core.tools import interpolate

def simple_mass_matrix(n_elements = 2, continuous = False):
    bf = basis_funcs.basis_from_degree(1)
    msh = simple_line_mesh(n_elements)
    q = quadrature.gauss(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", continuous, non_gen = True)
    init_dofs(msh)
    return assemble_mass_matrix(msh, q)

def test_mass_matrix():
    m = simple_mass_matrix()
    M_exact = np.array([[1.0 / 3.0, 1.0 / 6.0, 0, 0],
                        [1.0 / 6.0, 1.0 / 3.0, 0, 0],
                        [0, 0, 1.0 / 3.0, 1.0 / 6.0],
                        [0, 0, 1.0 / 6.0, 1.0 / 3.0]])
    np.testing.assert_almost_equal(M_exact, m[0:4, 0:4])

def test_mass_matrix_continuous():
    m = simple_mass_matrix(continuous = True)
    M_exact = np.array([[1.0 / 3.0, 1.0 / 6.0, 0],
                        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        [0, 1.0 / 6.0, 1.0 / 3.0]])
    np.testing.assert_almost_equal(M_exact, m[0:3, 0:3])

def test_mass_matrix_rhs():
    m = simple_mass_matrix()
    M_exact = np.array([[1.0 / 3.0, 1.0 / 6.0, 0, 0],
                        [1.0 / 6.0, 1.0 / 3.0, 0, 0],
                        [0, 0, 1.0 / 3.0, 1.0 / 6.0],
                        [0, 0, 1.0 / 6.0, 1.0 / 3.0]])
    rhs = mass_matrix_for_rhs(m)
    np.testing.assert_almost_equal(rhs[0:4], np.sum(M_exact, axis = 1))

def test_mass_matrix_functional():
    bf = basis_funcs.basis_from_nodes([0.001, 0.999])
    msh = simple_line_mesh(2)
    q = quadrature.gauss(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)

    fnc_coeffs = interpolate(lambda x,d: [[1,2][x[0] >= 0]] * 2, msh)
    apply_coeffs(msh, fnc_coeffs, "function_yay")
    basis_grabber = lambda e: e.function_yay

    m = mass_matrix_for_rhs(assemble_mass_matrix(msh, q, basis_grabber))
    M_exact = [0.5, 0.5, 1.0, 1.0]
    np.testing.assert_almost_equal(M_exact, m[0:4])
