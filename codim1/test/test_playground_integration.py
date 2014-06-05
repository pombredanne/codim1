import numpy as np
from codim1.core import *
from codim1.fast_lib import aligned_single_integral, TractionKernel,\
                            single_integral, ConstantBasis, HypersingularKernel,\
                            DisplacementKernel

def test_basis_quad_aligned_ordering():
    bf = gll_basis(9)
    quad_info = rl_quad(10, 0, -5, 1)
    np.testing.assert_almost_equal(np.array(bf.nodes), np.array(quad_info.x))

def test_fast_lobatto():
    N = 15
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    kernel = TractionKernel(1.0, 0.25)
    bf = gll_basis(N)
    one = ConstantBasis(np.ones(2))
    quad_info_old = lobatto(N + 1)
    pt = [0.0, -5.0]
    normal = [0.0, 1.0]

    kernel.set_interior_data(pt, normal)
    est_slow = single_integral(mapping.eval,
                              kernel, one, bf, quad_info_old, 0, 0)
    est_fast = aligned_single_integral(mapping.eval, kernel, bf,
                                     quad_info_old, 0)
    np.testing.assert_almost_equal(np.array(est_slow), np.array(est_fast))

def test_rl_integral():
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    kernel = TractionKernel(1.0, 0.25)
    bf = gll_basis(4)
    one = ConstantBasis(np.ones(2))

    quad_info_old = lobatto(5)
    x_bf = np.array(bf.nodes)
    x_q = np.array(quad_info_old.x)
    distance = 5.0
    quad_info_new = rl_quad(5, 0.0, distance, 5)
    # This one is comparing lobatto quadrature and recursive legendre quadrature
    # on the TractionKernel which is 1/r singular.

    pt = [0.0, -distance]
    normal = [0.0, 1.0]

    # exact = 0.2473475767
    exact = 0.00053055607635

    kernel.set_interior_data(pt, normal)
    est_gauss = single_integral(mapping.eval,
                              kernel, one, bf, quad_info_old, 0, 0)
    np.testing.assert_almost_equal(est_gauss[0][0], exact)

    est_gauss_fast = aligned_single_integral(mapping.eval, kernel, bf,
                                     quad_info_old, 0)
    np.testing.assert_almost_equal(est_gauss_fast[0][0], exact)

    # This stuff doesn't work yet
    est_new = single_integral(mapping.eval,
                              kernel, one, bf, quad_info_new, 0, 0)
    np.testing.assert_almost_equal(est_new[0][0], exact, 6)

    est_new_fast = aligned_single_integral(mapping.eval, kernel, bf,
                                     quad_info_new, 0)
    np.testing.assert_almost_equal(est_new_fast[0][0], exact, 6)

def test_gauss_displacement_xy():
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    kernel = DisplacementKernel(1.0, 0.25)
    degree = 1
    bf = gll_basis(degree)
    one = ConstantBasis(np.ones(2))
    quad_info_exact = gauss(100)
    quad_info = lobatto(degree + 2)

    # Testing GLL basis and quadrature combination on the nonsingular part
    # of the displacement kernel.
    pt = [0.5, 1000.0]
    normal = [0.0, 1.0]
    kernel.set_interior_data(pt, normal)

    integrate = lambda qi: single_integral(mapping.eval, kernel, one, bf, qi, 0, 0)
    exact = integrate(quad_info_exact)
    est = integrate(quad_info)
    np.testing.assert_almost_equal(exact[0][1], est[0][1], 16)