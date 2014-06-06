import numpy as np
from codim1.core import *
from codim1.fast_lib import aligned_single_integral, TractionKernel,\
                            single_integral, ConstantBasis,\
                            HypersingularKernel, DisplacementKernel,\
                            Kernel, QuadratureInfo
from quadracheer import legendre
from quadracheer import modify_divide_r2, mu_2_0, mu_2_1,\
        mu_4_0, mu_4_1, mu_6_0, mu_6_1, modify_times_x_minus_a, \
        recursive_quad, map_pts_wts, map_weights_by_inv_power,\
        map_singular_pt, map_distance_to_interval

def test_basis_quad_aligned_ordering():
    bf = gll_basis(9)
    quad_info = rl_quad(10, 0, -5)
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
    quad_info_new = rl_quad(5, 0.0, distance)
    # This one is comparing lobatto quadrature and recursive legendre quadrature
    # on the TractionKernel which is 1/r singular.

    pt = [0.0, -distance]
    normal = [0.0, 1.0]

    # exact = 0.2473475767
    exact = 0.00053055607635

    integrate = lambda qi: single_integral(mapping.eval,
                              kernel, one, bf, qi, 0, 0)
    aligned_integrate = lambda qi: \
        aligned_single_integral(mapping.eval, kernel, bf, qi, 0)

    kernel.set_interior_data(pt, normal)
    est_gauss = integrate(quad_info_old)
    np.testing.assert_almost_equal(est_gauss[0][0], exact)
    est_gauss_fast = aligned_integrate(quad_info_old)
    np.testing.assert_almost_equal(est_gauss_fast[0][0], exact)

    # This stuff doesn't work yet
    est_new = integrate(quad_info_new)
    np.testing.assert_almost_equal(est_new[0][0], exact, 6)

    est_new_fast = aligned_integrate(quad_info_new)
    np.testing.assert_almost_equal(est_new_fast[0][0], exact, 6)

class HypersingularPart1V1(Kernel):
    def _call(self, d, p, q):
        # print (d.dr[0] * d.dist) / (d.dr[1] * d.dist)
        return d.dr[p] * d.dr[q] * d.drdm * d.drdn / (d.dist ** 2)
class HypersingularPart1V2(Kernel):
    def _call(self, d, p, q):
        return 1.0# d.dr[p] * d.dr[q] * d.drdm * d.drdn / (d.dist ** 2)

def test_improved_hypersingular():
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    degree = 7
    bf = gll_basis(degree)
    one = ConstantBasis(np.ones(2))

    quad_info_exact = gauss(100)

    a = 0.01
    b = 0.05

    mapped_ay = map_singular_pt(a, 0.0, 1.0)
    mapped_by = map_distance_to_interval(b, 0.0, 1.0)
    quad_deg = degree + 4

    moments_xa0_r0 = legendre.legendre_integrals(quad_deg)
    moments_xa0_r2 = modify_divide_r2(quad_deg, moments_xa0_r0,
                    mapped_ay, mapped_by,
                    mu_2_0(mapped_ay, mapped_by), mu_2_1(mapped_ay, mapped_by))
    moments_xa0_r4 = modify_divide_r2(quad_deg, moments_xa0_r2,
                    mapped_ay, mapped_by,
                    mu_4_0(mapped_ay, mapped_by), mu_4_1(mapped_ay, mapped_by))
    moments_xa0_r6 = modify_divide_r2(quad_deg, moments_xa0_r4,
                    mapped_ay, mapped_by,
                    mu_6_0(mapped_ay, mapped_by), mu_6_1(mapped_ay, mapped_by))
    moments_xa1_r6 = modify_times_x_minus_a(len(moments_xa0_r6) - 2,
                                            moments_xa0_r6, mapped_ay)
    moments_xa2_r6 = modify_times_x_minus_a(len(moments_xa1_r6) - 2,
                                            moments_xa1_r6, mapped_ay)
    moments_xa3_r6 = modify_times_x_minus_a(len(moments_xa2_r6) - 2,
                                            moments_xa2_r6, mapped_ay)
    moments_xa4_r6 = modify_times_x_minus_a(len(moments_xa3_r6) - 2,
                                            moments_xa3_r6, mapped_ay)

    est = [[0, 0], [0, 0]]
    x, w = recursive_quad(moments_xa0_r6[:degree + 1])
    x, w = map_pts_wts(x, w, 0.0, 1.0)
    w = map_weights_by_inv_power(w, 6.0, 0.0, 1.0) * ((-b / 2) ** 2) * ((-2 * b)**2)
    est[1][1] = w[0]

    x, w = recursive_quad(moments_xa1_r6[:degree + 1])
    x, w = map_pts_wts(x, w, 0.0, 1.0)
    w = map_weights_by_inv_power(w, 6.0, 0.0, 1.0) * ((-b / 2) ** 2) * ((-2 * b)**1)
    est[0][1] = w[0]
    est[1][0] = w[0]
    x, w = recursive_quad(moments_xa2_r6[:degree + 1])
    x, w = map_pts_wts(x, w, 0.0, 1.0)
    w = map_weights_by_inv_power(w, 6.0, 0.0, 1.0) * ((-b / 2) ** 2) * ((-2 * b)**0)
    est[0][0] = w[0]

    # qi = QuadratureInfo(a, x, w)

    kernelv1 = HypersingularPart1V1()
    kernelv2 = HypersingularPart1V2()

    pt = [a, b]
    normal = [0.0, 1.0]
    kernelv1.set_interior_data(pt, normal)
    kernelv2.set_interior_data(pt, normal)

    integrate = lambda qi, k: single_integral(mapping.eval, k, one,
                                              bf, qi, 0, 0)
    exact = integrate(quad_info_exact, kernelv1)
    np.testing.assert_almost_equal(np.array(exact), np.array(est), 10)


def test_gauss_displacement_xy():
    mesh = simple_line_mesh(1, (0, 0), (1, 0))
    mapping = PolynomialMapping(mesh.elements[0])
    kernel = TractionKernel(1.0, 0.25)
    degree = 4
    bf = gll_basis(degree)
    one = ConstantBasis(np.ones(2))
    quad_info_exact = gauss(100)
    quad_info = lobatto(degree + 1)

    # Testing GLL basis and quadrature combination on the nonsingular part
    # of the displacement kernel.
    pt = [0.5, 1000.0]
    normal = [0.0, 1.0]
    kernel.set_interior_data(pt, normal)

    integrate = lambda qi: single_integral(mapping.eval, kernel, one, bf, qi, 0, 0)
    exact = integrate(quad_info_exact)
    est = integrate(quad_info)
    np.testing.assert_almost_equal(exact[0][1], est[0][1], 16)
