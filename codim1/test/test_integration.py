import numpy as np
import codim1.core.mesh as mesh
import codim1.core.quad_strategy as quad_strategy
import codim1.core.quadrature as quadrature
import codim1.core.basis_funcs as basis_funcs
from codim1.fast_lib import TractionKernel, DisplacementKernel
from codim1.fast.integration import double_integral
from codim1.fast.integration import single_integral

def test_exact_dbl_integrals_H_same_element():
    msh = mesh.Mesh.simple_line_mesh(1)
    qs = quad_strategy.QuadStrategy(msh, 10, 10, 10, 10)
    bf = basis_funcs.BasisFunctions.from_degree(1)

    kernel = TractionKernel(1.0, 0.25)
    H_00 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_oneoverr, 0, 0, 0, 0)
    np.testing.assert_almost_equal(H_00, np.zeros((2, 2)), 3)

    H_11 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_oneoverr, 0, 1, 0, 1)
    np.testing.assert_almost_equal(H_11, np.zeros((2, 2)), 3)

    H_01 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_oneoverr, 0, 0, 0, 1)
    H_01_exact = np.array([[0.0, 1 / (6 * np.pi)],
                           [-1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_01, H_01_exact, 3)

    H_10 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_oneoverr, 0, 1, 0, 0)
    H_10_exact = np.array([[0.0, -1 / (6 * np.pi)],
                           [1 / (6 * np.pi), 0.0]])
    np.testing.assert_almost_equal(H_10, H_10_exact, 3)

def test_exact_dbl_integrals_G_same_element():
    msh = mesh.Mesh.simple_line_mesh(1)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    qs = quad_strategy.QuadStrategy(msh, 10, 10, 10, 10)
    kernel = DisplacementKernel(1.0, 0.25)
    G_00 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_logr, 0, 0, 0, 0)
    np.testing.assert_almost_equal(G_00, [[0.165187, 0], [0, 0.112136]], 4)
    G_10 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_logr, 0, 1, 0, 0)
    np.testing.assert_almost_equal(G_10, [[0.112136, 0], [0, 0.0590839]], 4)
    G_01 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_logr, 0, 0, 0, 1)
    np.testing.assert_almost_equal(G_01, [[0.112136, 0], [0, 0.0590839]], 4)
    G_11 = double_integral(msh, kernel, bf, bf, qs.get_simple(),
            qs.quad_logr, 0, 1, 0, 1)
    np.testing.assert_almost_equal(G_11, [[0.165187, 0], [0, 0.112136]], 4)

def test_exact_dbl_integrals_G_different_element():
    msh = mesh.Mesh.simple_line_mesh(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    qs = quad_strategy.QuadStrategy(msh, 10, 10, 10, 10)
    kernel = DisplacementKernel(1.0, 0.25)
    q = [qs.quad_shared_edge_left] * \
            len(qs.get_simple().x)
    G_00 = double_integral(msh, kernel, bf, bf, qs.get_simple(), q, 0, 0, 1, 0)
    np.testing.assert_almost_equal(G_00, [[0.0150739, 0], [0, 0.00181103]], 4)
    G_10 = double_integral(msh, kernel, bf, bf, qs.get_simple(), q, 0, 1, 1, 0)
    np.testing.assert_almost_equal(G_10,
            [[0.02833119, 0], [0, 0.01506828]], 4)
    G_01 = double_integral(msh, kernel, bf, bf, qs.get_simple(), q, 0, 0, 1, 1)
    np.testing.assert_almost_equal(G_01,
            [[0.00663146, 0], [0, -0.00663146]], 4)
    G_11 = double_integral(msh, kernel, bf, bf, qs.get_simple(), q, 0, 1, 1, 1)
    np.testing.assert_almost_equal(G_11, [[0.0150739, 0], [0, 0.00181103]], 4)

def test_realistic_double_integral_symmetry():
    msh = mesh.Mesh.simple_line_mesh(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    qs = quad_strategy.QuadStrategy(msh, 10, 10, 10, 10)
    kernel = DisplacementKernel(1.0, 0.25)

    # fnc = lambda r, n: 1 / r[0]
    one = double_integral(msh, kernel, bf, bf, qs.get_simple(),
                           qs.quad_logr, 1, 0, 1, 1)

    two = double_integral(msh, kernel, bf, bf, qs.get_simple(),
                           qs.quad_logr, 1, 1, 1, 0)
    np.testing.assert_almost_equal(one, two)


def test_M_integral_same_dof():
    msh = mesh.Mesh.simple_line_mesh(2)
    q = quadrature.QuadGauss(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    kernel = lambda x, n: np.array([[1.0, 0], [0, 1.0]])
    M_local = single_integral(msh, kernel, bf, bf, q, 0, 0, 0)
    # integral of (1-x)^2 from 0 to 1
    np.testing.assert_almost_equal(M_local[0, 0], 1.0 / 3.0)

def test_M_integral_same_dof_with_jacobian():
    msh = mesh.Mesh.simple_line_mesh(4)
    q = quadrature.QuadGauss(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    kernel = lambda x, n: np.array([[1.0, 0], [0, 1.0]])
    M_local = single_integral(msh, kernel, bf, bf, q, 0, 0, 0)
    # Element size divided by two so the M value should be divided by two
    np.testing.assert_almost_equal(M_local[0, 0], 1.0 / 6.0)

def test_M_integral_diff_dof():
    msh = mesh.Mesh.simple_line_mesh(2)
    q = quadrature.QuadGauss(2)
    bf = basis_funcs.BasisFunctions.from_degree(1)
    kernel = lambda x, n: np.array([[1.0, 0], [0, 1.0]])
    M_local = single_integral(msh, kernel, bf, bf, q, 0, 0, 1)
    # integral of (1-x)*x from 0 to 1
    np.testing.assert_almost_equal(M_local[0, 0], 1.0 / 6.0)
