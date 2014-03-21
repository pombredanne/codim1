import numpy as np
from codim1.core.mass_matrix import MassMatrix
import codim1.core.mesh as mesh
import codim1.core.basis_funcs as basis_funcs
import codim1.core.dof_handler as dof_handler
import codim1.core.quadrature as quadrature

def simple_mass_matrix(n_elements = 2):
    msh = mesh.Mesh.simple_line_mesh(n_elements)
    q = quadrature.QuadGauss(2)
    dh = dof_handler.DiscontinuousDOFHandler(msh, 1)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    m = MassMatrix(msh, bf, bf, dh, q)
    return m

def test_mass_matrix():
    m = simple_mass_matrix()
    m.compute()
    M_exact = np.array([[1.0 / 3.0, 1.0 / 6.0, 0, 0],
                        [1.0 / 6.0, 1.0 / 3.0, 0, 0],
                        [0, 0, 1.0 / 3.0, 1.0 / 6.0],
                        [0, 0, 1.0 / 6.0, 1.0 / 3.0]])
    np.testing.assert_almost_equal(M_exact, m.M[0:4, 0:4])

