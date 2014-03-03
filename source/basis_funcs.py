import numpy as np
import scipy.interpolate as spi
import copy

class BasisFunctions(object):
    """
        This class handles interactions with Lagrange polynomials defined on
        the unit reference interval [0, 1].
    """

    @classmethod
    def from_degree(cls, element_deg):
        """
            Create an equispaced nodal basis.
        """
        if element_deg == 0:
            nodes = np.array([0.5])
        else:
            nodes = np.linspace(0.0, 1.0, element_deg + 1)
        return cls(nodes)

    def __init__(self, nodes):
        """
            Builds the Lagrange interpolating polynomials with nodes at the
            points specified.
        """
        self.fncs = []
        self.nodes = copy.copy(nodes)
        self.num_fncs = len(nodes)
        for (i, n) in enumerate(nodes):
            w = np.zeros_like(nodes)
            w[i] = 1.0
            # scipy.interpolate.lagrange has trouble above 20 nodes, but that
            # shouldn't be an issue for this code
            poly = spi.lagrange(nodes, w)
            self.fncs.append(poly)

    def evaluate_basis(self, i, x):
        """
            Evaluates the i-th lagrange polynomial at x.
        """
        return self.fncs[i](x)

    def evaluate_basis_derivative(self, i, x):
        """
            Evaluates the derivative of the i-th lagrange polynomial at x
        """
        return self.fncs[i].deriv()(x)

    def times_coeffs(self, x, C):
        """
            Evaluates the sum of the lagrange polynomials times their
            respective coefficients -- C.
            C should be a numpy array
        """
        return C * np.array([f(x) for f in self.fncs])


################################################################################
# TESTS                                                                        #
################################################################################

def test_degree_zero():
    bf = BasisFunctions.from_degree(0)
    assert(bf.nodes[0] == 0.5)
    assert(len(bf.nodes) == 1)
    for x in np.linspace(0.0, 1.0, 10):
        np.testing.assert_almost_equal(bf.evaluate_basis(0, x), 1.0)
        np.testing.assert_almost_equal(bf.evaluate_basis_derivative(0, x), 0.0)


def test_from_degree():
    bf = BasisFunctions.from_degree(2)
    assert(bf.nodes[0] == 0.0)
    assert(bf.nodes[1] == 0.5)
    assert(bf.nodes[2] == 1.0)

def test_basis():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    # Check to make sure the nodes are right
    np.testing.assert_almost_equal(bf.evaluate_basis(0, -1.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(0, 0.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(0, 1.0), 0.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(1, 0.0), 1.0)
    np.testing.assert_almost_equal(bf.evaluate_basis(2, 1.0), 1.0)


    # Test for 2nd degree polynomials
    x_hat = 0.3
    y_exact1 = 0.5 * x_hat * (x_hat - 1)
    y_exact2 = (1 - x_hat) * (1 + x_hat)
    y_exact3 = 0.5 * x_hat * (1 + x_hat)
    y_est1 = bf.evaluate_basis(0, x_hat)
    y_est2 = bf.evaluate_basis(1, x_hat)
    y_est3 = bf.evaluate_basis(2, x_hat)
    np.testing.assert_almost_equal(y_exact1, y_est1)
    np.testing.assert_almost_equal(y_exact2, y_est2)
    np.testing.assert_almost_equal(y_exact3, y_est3)

    C = np.array([0.5, 0.7, -0.1])
    exact = C * [y_exact1, y_exact2, y_exact3]
    np.testing.assert_almost_equal(exact, bf.times_coeffs(x_hat, C))

def test_basis_derivative():
    bf = BasisFunctions([-1.0, 0.0, 1.0])
    x_hat = 0.3
    yd_exact1 = x_hat - 0.5
    yd_exact2 = -2 * x_hat
    yd_exact3 = x_hat + 0.5
    yd_est1 = bf.evaluate_basis_derivative(0, x_hat)
    yd_est2 = bf.evaluate_basis_derivative(1, x_hat)
    yd_est3 = bf.evaluate_basis_derivative(2, x_hat)
    np.testing.assert_almost_equal(yd_exact1, yd_est1)
    np.testing.assert_almost_equal(yd_exact2, yd_est2)
    np.testing.assert_almost_equal(yd_exact3, yd_est3)
