import numpy as np
from codim1.core.basis_funcs import BasisFunctions, Solution
from codim1.core.mesh import Mesh
import codim1.core.dof_handler as dof_handler

def test_degree_zero():
    bf = BasisFunctions.from_degree(0)
    assert(bf.nodes[0] == 0.5)
    assert(len(bf.nodes) == 1)
    for x in np.linspace(0.0, 1.0, 10):
        np.testing.assert_almost_equal(bf.evaluate(0, 0, x, x), 1.0)
        np.testing.assert_almost_equal(
                bf.evaluate_derivative(0, 0, x, x), 0.0)

def test_from_degree():
    bf = BasisFunctions.from_degree(2)
    assert(bf.nodes[0] == 0.0)
    assert(bf.nodes[1] == 0.5)
    assert(bf.nodes[2] == 1.0)

def test_basis_nodes():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    # Check to make sure the nodes are right
    np.testing.assert_almost_equal(bf.evaluate(0, 0, -1.0, 1.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate(0, 0, 0.0, 1.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate(0, 0, 1.0, 1.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate(0, 1, 0.0, 1.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate(0, 2, 1.0, 1.0), 1.0)


def test_basis_2nd_order():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    # Test for 2nd degree polynomials
    x_hat = 0.3
    y_exact1 = 0.5 * x_hat * (x_hat - 1)
    y_exact2 = (1 - x_hat) * (1 + x_hat)
    y_exact3 = 0.5 * x_hat * (1 + x_hat)
    y_est1 = bf.evaluate(0, 0, x_hat, x_hat)
    y_est2 = bf.evaluate(0, 1, x_hat, x_hat)
    y_est3 = bf.evaluate(0, 2, x_hat, x_hat)
    np.testing.assert_almost_equal(y_exact1, y_est1)
    np.testing.assert_almost_equal(y_exact2, y_est2)
    np.testing.assert_almost_equal(y_exact3, y_est3)

def test_basis_derivative():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    x_hat = 0.3
    yd_exact1 = x_hat - 0.5
    yd_exact2 = -2 * x_hat
    yd_exact3 = x_hat + 0.5
    yd_est1 = bf.evaluate_derivative(0, 0, x_hat, x_hat)
    yd_est2 = bf.evaluate_derivative(0, 1, x_hat, x_hat)
    yd_est3 = bf.evaluate_derivative(0, 2, x_hat, x_hat)
    np.testing.assert_almost_equal(yd_exact1, yd_est1)
    np.testing.assert_almost_equal(yd_exact2, yd_est2)
    np.testing.assert_almost_equal(yd_exact3, yd_est3)

def test_function():
    f = lambda x: (1.0, x[1])
    F = BasisFunctions.from_function(f)
    f_val = F.evaluate(0.0, 0, 1.0, (0.0, 11.1))

    assert(f_val[0] == 1.0)
    assert(f_val[1] == 11.1)

def test_gradient_chainrule():
    msh = Mesh.simple_line_mesh(1)
    bf = BasisFunctions([0.0, 1.0])
    gradient = bf.get_gradient_basis(msh)
    assert(gradient.chain_rule(0) == 0.5)
    np.testing.assert_almost_equal(gradient.evaluate(0, 0, 0.5, 0.0),
                                   bf.evaluate_derivative(0, 0, 0.5, 0.0))
    assert(bf.chain_rule(0) == 1.0)

def test_solution():
    msh = Mesh.simple_line_mesh(1)
    dh = dof_handler.DiscontinuousDOFHandler(msh, 1)
    bf = BasisFunctions([0.0, 1.0])
    coeffs = np.array([0.0, 1.0, 1.0, 0.0])
    soln = Solution(bf, dh, coeffs)

    value = soln.evaluate(0, 0, 0.0, -1.0)
    assert(value[0] == 0.0)
    assert(value[1] == 1.0)

    value = soln.evaluate(0, 1, 1.0, -1.0)
    assert(value[0] == 1.0)
    assert(value[1] == 0.0)
