import numpy as np
import copy
import gaussian_quad

class QuadGauss(object):
    """
    Simple wrapper of gaussian quadrature. Returns points and weights for
    quadrature on [a, b]. Default is a = 0, b = 1.0
    """
    def __init__(self, N, a = 0.0, b = 1.0):
        self.N = N
        self.x, self.w = gaussian_quad.gaussxwab(self.N, a, b)


class QuadSingularTelles(object):
    """
    Use a cubic polynomial transformation to turn a 1D cauchy principal value
    integral into a easily integrable form.
    This should also be able to accurately integrate terms like
    log(|r|) or 1/r where r = (x - y).

    See
    "A SELF-ADAPTIVE CO-ORDINATE TRANSFORMATION FOR EFFICIENT NUMERICAL
    EVALUATION OF GENERAL BOUNDARY ELEMENT INTEGRALS", Telles, 1987.
    for a description of the method. I use the same notation adopted in that
    paper. Because the reference segment is [0, 1] here and the reference
    segment is [-1, 1] in the Telles paper, a small extra transformation
    is performed.

    Note there is a printing error in the Jacobian of the transformation
    from gamma coordinates to eta coordinates in the Telles paper. The
    formula in the paper is
    (3 * (gamma - gamma_bar ** 2)) / (1 + 3 * gamma_bar ** 2)
    It SHOULD read:
    (3 * (gamma - gamma_bar) ** 2) / (1 + 3 * gamma_bar ** 2)
    """
    def __init__(self, N, x0):
        self.N = N
        self.x0 = x0

        # The location of the singularity in eta space
        eta_bar = 2 * x0 - 1.0

        eta_star = eta_bar ** 2 - 1.0

        # The location of the singularity in gamma space
        term1 = (eta_bar * eta_star + np.abs(eta_star))
        term2 = (eta_bar * eta_star - np.abs(eta_star))

        # Fractional powers of negative numbers are multiply valued and python
        # recognizes this. So, I specify that I want the real valued third root
        gamma_bar = np.sign(term1) * np.abs(term1) ** (1.0 / 3.0) + \
                np.sign(term2) * np.abs(term2) ** (1.0 / 3.0) + \
                eta_bar

        gauss_quadrature = QuadGauss(N, -1.0, 1.0)
        gamma = gauss_quadrature.x
        gamma_weights = gauss_quadrature.w
        self.x = ((gamma - gamma_bar) ** 3 + gamma_bar * (gamma_bar ** 2 + 3))\
                / (2 * (1 + 3 * gamma_bar ** 2))\
                + 0.5

        # If we accidentally choose a Gaussian integration scheme that
        # exactly sample the singularity, this method will fail. This can
        # be easily remedied by simply increasing the order of
        # For example, this happens if x0 == 0 and N is odd.
        if (self.x == x0).any():
            raise Exception("Telles integration has sampled the " +
                    "singularity. Choose a different order of integration.")

        self.w = gamma_weights * (3 * (gamma - gamma_bar) ** 2) \
                / (2 * (1 + 3 * gamma_bar ** 2))

class QuadOneOverR(object):
    """
    Quadrature points and weights for integrating a function with form
    f(x) / (x - x0)
    on the interval [0, 1]
    Uses the 2N point gauss rule derived in Piessens (1970) Almost certainly
    suboptimal, but it's very simple and it works. Exact for polynomials of
    order 4N.

    An alternative is the method by Longman (1958). It's significantly worse
    though... Exact for polynomials of order 2N.
    """
    def __init__(self, N, x0, nonsingular_N = -1):
        if nonsingular_N == -1:
            nonsingular_N = N
        self.nonsingular_N = nonsingular_N
        self.N = N
        self.x0 = x0

        # Split the interval into two sections. One is properly integrable.
        # The other is symmetric about the singularity point and must be
        # computed using as a cauchy principal value integral.
        if self.x0 < 0.5:
            proper_length = 1 - (2 * self.x0)
            pv_length = 2 * self.x0
            pv_start = 0.0
            proper_start = 2 * self.x0
        else:
            proper_start = 0.0
            pv_start = -1.0 + 2 * self.x0
            proper_length = -1.0 + 2 * self.x0
            pv_length = 2.0 - 2 * self.x0

        # Just check...
        assert(pv_length + proper_length == 1.0)
        assert(pv_start + pv_length == proper_start
            or pv_start + pv_length == 1.0)
        assert(proper_start + proper_length == pv_start
            or proper_start + proper_length == 1.0)
        assert(pv_start + pv_length / 2.0 == self.x0)


        # the interval without the singularity
        qg = QuadGauss(nonsingular_N, proper_start,
                        proper_start + proper_length)
        self.x = qg.x
        self.w = qg.w

        self.x_sing, self.w_sing = self.piessen_method(N, pv_start,
                                                pv_length, x0)

        # Finished!
        self.x = np.append(self.x, self.x_sing)
        self.w = np.append(self.w, self.w_sing)

    @staticmethod
    def longman_method(N, pv_start, pv_length):
        # # The interval with the singularity
        qg_sing1 = QuadGauss(N, pv_start + pv_length / 2.0,
                                pv_start + pv_length)
        x_sing = qg_sing1.x
        x_sing = np.append(x_sing,
                2 * pv_start + pv_length - qg_sing1.x)

        w_1 = qg_sing1.w# * (qg_sing1.x - x0)
        w_2 = qg_sing1.w# * (qg_sing1.x - x0)
        w_sing = w_1
        w_sing = np.append(w_sing, w_2)

    @staticmethod
    def piessen_method(N, pv_start, pv_length, x0, add_singularity = True):
        x_base, w_base = QuadOneOverR.piessen_neg_one_to_one_nodes(N)
        # Convert to the interval from [pv_start, pv_start + pv_length]
        x = (pv_length / 2) * x_base + \
                (2 * pv_start + pv_length) / 2.0
        # No need to scale the weights because the linear factor in the 1/r
        # exactly cancels the jacobian.
        w = w_base

        # If we don't factor out the 1 / (x - x0) of the quadratured function,
        # so we must account for it here.
        if add_singularity:
            w *= x - x0
        return x, w

    @staticmethod
    def piessen_neg_one_to_one_nodes(N):
        """Piessen nodes and weights for [-1, 1]"""
        if N % 2 == 1:
            raise Exception("Piessens method requires an even quadrature " +
                    "order")

        qg_sing1 = QuadGauss(2 * N, -1., 1.)
        x = qg_sing1.x
        w = qg_sing1.w / qg_sing1.x
        return x, w


################################################################################
# TESTS                                                                        #
################################################################################


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
    from elastic_kernel import ElastostaticKernel
    k = ElastostaticKernel(1.0, 0.25)
    f = lambda x, y: k.displacement_singular((x - y, 0.0), 0.0) * \
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

