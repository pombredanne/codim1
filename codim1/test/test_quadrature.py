import numpy as np
from codim1.core.quadrature import QuadGauss, QuadSingularTelles, QuadOneOverR
from codim1.fast.elastic_kernel import ElastostaticKernel
import codim1.core.gaussian_quad as gaussian_quad

################################################################################
# TESTS                                                                        #
################################################################################

def test_build_sets():
    x, w = gaussian_quad.gaussxwab(3, -1.0, 1.0)
    x2, w2 = gaussian_quad.gaussxwab(12, -1.0, 1.0)
    np.testing.assert_almost_equal(x, gaussian_quad.x_set[3])
    np.testing.assert_almost_equal(w, gaussian_quad.w_set[3])
    np.testing.assert_almost_equal(x2, gaussian_quad.x_set[12])
    np.testing.assert_almost_equal(w2, gaussian_quad.w_set[12])

def test_gaussxw():
    x, w = gaussian_quad.gaussxwab(3, -1.0, 1.0)
    # Exact values retrieved from the wikipedia page on Gaussian Quadrature
    # The main function has been tested by the original author for a wide
    # range of orders. But, this is just to check everything is still working
    # properly
    np.testing.assert_almost_equal(x[0], np.sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(x[1], 0.0)
    np.testing.assert_almost_equal(x[2], -np.sqrt(3.0 / 5.0))
    np.testing.assert_almost_equal(w[0], 5.0 / 9.0)
    np.testing.assert_almost_equal(w[1], 8.0 / 9.0)
    np.testing.assert_almost_equal(w[2], 5.0 / 9.0)

def test_QuadGauss():
    f = lambda x: 3 * x ** 2
    F = lambda x: x ** 3
    exact = F(1) - F(0)
    q = QuadGauss(2)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est)

def test_QuadTellesPaper():
    g = lambda x: np.log(np.abs(0.3 + x))
    f = lambda y: 2 * g(2 * y - 1)
    exact = -1.908598917
    # We compare with the result Telles gets for 10 points to make sure
    # the method is properly implemented.
    telles_paper_10_pts = -1.90328
    q = QuadSingularTelles(10, 0.35)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(telles_paper_10_pts, est, 5)

def test_QuadLogR():
    f = lambda x: np.log(np.abs(x - 0.5))
    exact = -1.0 - np.log(2.0)
    q = QuadSingularTelles(50, 0.5)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est, 4)

def test_QuadLogR2():
    f = lambda x: x ** 2 * np.log(np.abs(x - 0.9))
    exact = -0.764714
    q = QuadSingularTelles(40, 0.9)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est, 4)

def test_anotherLogRDouble_G11_from_kernel():
    k = ElastostaticKernel(1.0, 0.25)
    f = lambda x, y: k.displacement_kernel(x - y, 0.0, 0.0, 0.0) * \
            (1 - x) * (1 - y)

    exact = 7 / (48 * np.pi)
    q = QuadGauss(75)
    sum = 0.0
    for (pt, wt) in zip(q.x, q.w):
        q_inner = QuadSingularTelles(76, pt)
        g = lambda x: f(x, pt)
        for (inner_pt, inner_wt) in zip(q_inner.x, q_inner.w):
            sum += g(inner_pt)[1, 1] * inner_wt * wt
    np.testing.assert_almost_equal(exact, sum)

def test_anotherLogRDouble_G11():
    f = lambda x, y: (1 / (3 * np.pi)) *\
        np.log(1.0 / np.abs(x - y)) * x * y
    exact = 7 / (48 * np.pi)
    q = QuadGauss(75)
    sum = 0.0
    for (pt, wt) in zip(q.x, q.w):
        q_inner = QuadSingularTelles(76, pt)
        g = lambda x: f(x, pt)
        sum += np.sum(g(q_inner.x) * q_inner.w * wt)
    np.testing.assert_almost_equal(exact, sum)

def test_piessen_neg_1_1():
    # Example 1 from Piessens
    f = lambda x: np.exp(x)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = QuadOneOverR.piessen_neg_one_to_one_nodes(2)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1():
    # Example 1 from Piessens mapped to [0,1]
    g = lambda x: np.exp(x)
    f = lambda x: g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = QuadOneOverR.piessen_method(2, 0.0, 1.0, 0.5, False)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1_with_singularity():
    # Example 1 from Piessens mapped to [0,1] and with singularity
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = QuadOneOverR.piessen_method(2, 0.0, 1.0, 0.5)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_QuadOneOverR_1():
    f = lambda x: 1 / (x - 0.4)
    exact = np.log(3.0 / 2.0)
    q = QuadOneOverR(2, 0.4, nonsingular_N = 10)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_2():
    # Example 1 from Piessens
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    q = QuadOneOverR(8, 0.5)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_3():
    # Example 2 from Piessens
    g = lambda x: np.exp(x) / (np.sin(x) - np.cos(x))
    f = lambda x: np.pi / 2.0 * g(np.pi / 2.0 * x)
    exact = 2.61398312
    # Piessens estimate derived with a two pt rule.
    piessens_est = 2.61398135
    q = QuadOneOverR(2, 0.5)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(piessens_est, est)

# Tests x in the upper half of the interval
def test_QuadOneOverR_4():
    f = lambda x: np.exp(x) / (x - 0.8)
    exact = -1.13761642399
    q = QuadOneOverR(2, 0.8, nonsingular_N = 20)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est)

# Tests x in the lower half of the interval.
def test_QuadOneOverR_5():
    f = lambda x: np.exp(x) / (x - 0.2)
    exact = 3.139062607254266
    q = QuadOneOverR(2, 0.2, nonsingular_N = 50)
    est = np.sum(f(q.x) * q.w)
    np.testing.assert_almost_equal(exact, est)

