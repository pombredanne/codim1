from math import sqrt
import numpy as np
from codim1.core.quadrature import gauss, piessens, telles_singular,\
        telles_quasi_singular, rl1_quad
from codim1.core.basis_funcs import gll_basis

def test_gauss():
    qi = gauss(3)
    np.testing.assert_almost_equal(qi.x[0], sqrt(3.0 / 5.0) * 0.5 + 0.5)

def test_QuadOneOverR_1():
    f = lambda x: 1 / (x - 0.4)
    exact = np.log(3.0 / 2.0)
    qi = piessens(2, 0.4, nonsingular_N = 10)
    qx = np.array(qi.x)
    qw = np.array(qi.w)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_quadlogr2():
    f = lambda x: x ** 2 * np.log(np.abs(x - 0.9))
    exact = -0.764714
    qi = telles_singular(40, 0.9)
    qx = np.array(qi.x)
    qw = np.array(qi.w)
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est, 4)

def test_quasi_singular():
    sing_pt = 1.004
    g = lambda x: (sing_pt - x) ** -2
    exact = 249.003984064
    x_nearest = 1.0
    D = sing_pt - 1.0
    N = 35
    qi = telles_quasi_singular(N, x_nearest, D)
    qx = np.array(qi.x)
    qw = np.array(qi.w)
    est = np.sum(g(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_rl1_quad():
    N = 5
    values = [(0.2, 0.3, 0.2949533988361775),
              (1.2, 4.0, 0.04976184140700821),
              (1.5, 1.5, 0.1216505480495554)]
    for (ay, by, exact) in values:
        qi = rl1_quad(N, ay, by)
        qx = np.array(qi.x)
        qw = np.array(qi.w)

        gll_nodes = gll_basis(N).nodes
        np.testing.assert_almost_equal(gll_nodes, qx)

        f = lambda x: x ** 4
        est = np.sum(f(qx) * qw)
        # DUDE 13 digits!
        np.testing.assert_almost_equal(est, exact, 13)

if __name__ == "__main__":
    test_gauss()
