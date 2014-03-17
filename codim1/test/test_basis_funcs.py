import numpy as np
from codim1.core.basis_funcs import BasisFunctions

def test_degree_zero():
    bf = BasisFunctions.from_degree(0)
    assert(bf.nodes[0] == 0.5)
    assert(len(bf.nodes) == 1)
    for x in np.linspace(0.0, 1.0, 10):
        np.testing.assert_almost_equal(bf.evaluate_basis(0, x, x), 1.0)
        np.testing.assert_almost_equal(
                bf.derivs.evaluate_basis(0, x, x), 0.0)

def test_from_degree():
    bf = BasisFunctions.from_degree(2)
    assert(bf.nodes[0] == 0.0)
    assert(bf.nodes[1] == 0.5)
    assert(bf.nodes[2] == 1.0)

def test_basis():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    # Check to make sure the nodes are right
    np.testing.assert_almost_equal(bf.evaluate_basis(0, -1.0, 1.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(0, 0.0, 1.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(0, 1.0, 1.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(1, 0.0, 1.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(2, 1.0, 1.0), 1.0)


    # Test for 2nd degree polynomials
    x_hat = 0.3
    y_exact1 = 0.5 * x_hat * (x_hat - 1)
    y_exact2 = (1 - x_hat) * (1 + x_hat)
    y_exact3 = 0.5 * x_hat * (1 + x_hat)
    y_est1 = bf.evaluate_basis(0, x_hat, x_hat)
    y_est2 = bf.evaluate_basis(1, x_hat, x_hat)
    y_est3 = bf.evaluate_basis(2, x_hat, x_hat)
    np.testing.assert_almost_equal(y_exact1, y_est1)
    np.testing.assert_almost_equal(y_exact2, y_est2)
    np.testing.assert_almost_equal(y_exact3, y_est3)

def test_basis_derivative():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    derivs = bf.derivs
    x_hat = 0.3
    yd_exact1 = x_hat - 0.5
    yd_exact2 = -2 * x_hat
    yd_exact3 = x_hat + 0.5
    yd_est1 = derivs.evaluate_basis(0, x_hat, x_hat)
    yd_est2 = derivs.evaluate_basis(1, x_hat, x_hat)
    yd_est3 = derivs.evaluate_basis(2, x_hat, x_hat)
    np.testing.assert_almost_equal(yd_exact1, yd_est1)
    np.testing.assert_almost_equal(yd_exact2, yd_est2)
    np.testing.assert_almost_equal(yd_exact3, yd_est3)

def test_function():
    f = lambda x: (1.0, x[1])
    F = BasisFunctions.from_function(f)
    f_val = F.evaluate_basis(0, 1.0, (0.0, 11.1))

    assert(f_val[0] == 1.0)
    assert(f_val[1] == 11.1)
