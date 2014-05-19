import numpy as np
from codim1.assembly import mass_matrix_for_rhs, assemble_mass_matrix
from codim1.core import *
import codim1.core.basis_funcs as basis_funcs
import codim1.core.dof_handler as dof_handler
import codim1.core.quadrature as quadrature

def simple_mass_matrix(n_elements = 2, continuous = False):
    bf = basis_funcs.BasisFunctions.from_degree(1)
    msh = simple_line_mesh(n_elements)
    q = quadrature.QuadGauss(2)
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
    np.testing.assert_almost_equal(mass_matrix_for_rhs(m)[0:4],
                                   np.sum(M_exact, axis = 1))

def test_mass_matrix_functional():
    bf = basis_funcs.BasisFunctions.from_degree(1)
    msh = simple_line_mesh(2)
    q = quadrature.QuadGauss(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)

    fnc = BasisFunctions.from_function(lambda x,d: [1,2][x[0] > 0])
    basis_grabber = lambda e: fnc
    m = assemble_mass_matrix(msh, q, basis_grabber)
    M_exact = np.zeros((4,4))
    M_exact[0,0] = 0.5
    M_exact[1,0] = 0.5
    M_exact[2,2] = 1
    M_exact[3,2] = 1
    np.testing.assert_almost_equal(M_exact, m[0:4, 0:4])
