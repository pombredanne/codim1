from codim1.fast.elastic_kernel import *
import numpy as np

def test_traction_kernel_elements():
    E = 1e5
    nu = 0.3
    shear_modulus = E / (2 * (1 + nu))
    kernel = TractionKernel(shear_modulus, nu)
    T = kernel.call(np.array([0, 4.7285]),
                               np.zeros(2),
                               np.array([-1.0, 0.0]))
    exact = np.array([[0, 0.0096],[-0.0096, 0]])
    np.testing.assert_almost_equal(exact, T, 4)

def test_displacement_symmetry():
    kernel = DisplacementKernel(1.0, 0.25)
    a = kernel.call(np.array([1.0, 0.5]),
                                   np.array([0.0, 0.0]),
                                   np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(a - a.T, np.zeros_like(a))

def test_displacement_mirror_symmetry():
    kernel = DisplacementKernel(1.0, 0.25)
    a = kernel.call(np.array([1.0, 0.5]),
                    np.zeros(2), np.array([1.0, 0.0]))
    b = kernel.call(np.array([-1.0, -0.5]),
                    np.zeros(2), np.array([1.0, 0.0]))
    np.testing.assert_almost_equal(a, b)

def test_traction_mirror_symmety():
    kernel = TractionKernel(1.0, 0.25)
    a = kernel.call(np.array([1.0, 0.5]),
                    np.zeros(2), np.array([1.0, 0.0]))
    # Only symmetric if we reverse the normal vector too!
    b = kernel.call(np.array([-1.0, -0.5]),
                    np.zeros(2), np.array([-1.0, 0.0]))
    np.testing.assert_almost_equal(a, b)

def test_displacement():
    kernel = DisplacementKernel(1.0, 0.25)
    G = kernel.call(np.array([2.0, 0.0]),
                    np.array([0, 0.0]),
                    np.array([0, 1.0]))
    np.testing.assert_almost_equal(G[0, 0],
                                    (2 * np.log(1 / 2.0) + 1) / (6 * np.pi))
    np.testing.assert_almost_equal(G[1, 0], 0.0)
    np.testing.assert_almost_equal(G[0, 1], 0.0)
    np.testing.assert_almost_equal(G[1, 1],
                                    (2 * np.log(1 / 2.0)) / (6 * np.pi))

def test_traction():
    kernel = TractionKernel(1.0, 0.25)
    H = kernel.call(np.array([2.0, 0.0]),
                    np.array([0, 0.0]),
                    np.array([0, 1.0]))
    np.testing.assert_almost_equal(H[0, 1],
                                    1 / (6 * np.pi * 2.0))
    np.testing.assert_almost_equal(H[0, 0], 0.0)
    np.testing.assert_almost_equal(H[1, 1], 0.0)
    np.testing.assert_almost_equal(H[1, 0], -H[0, 1])

def test_traction_adjoint():
    kernel = AdjointTractionKernel(1.0, 0.25)
    HT = kernel.call(np.array([2.0, 0.0]),
                    np.array([0, 1.0]),
                    np.array([0, 0.0]))
    np.testing.assert_almost_equal(HT[0, 1],
                                    1 / (6 * np.pi * 2.0))
    np.testing.assert_almost_equal(HT[0, 0], 0.0)
    np.testing.assert_almost_equal(HT[1, 1], 0.0)
    np.testing.assert_almost_equal(HT[1, 0], -HT[0, 1])

def test_hypersingular_regularized():
    kernel = RegularizedHypersingularKernel(1.0, 0.25)
    W = kernel.call(np.array([2.0, 0.0]),
                    np.array([0, 1.0]),
                    np.array([0, 0.0]))
    W_exact = np.array([[2 * (np.log(2) + 1) / (3 * np.pi), 0],
                        [0, 2 * np.log(2) / (3 * np.pi)]])

def test_hypersingular_nonregularized():
    kernel = HypersingularKernel(1.0, 0.25)
    S = kernel.call(np.array([2.0, 0.0]),
                    np.array([1, 0.0]),
                    np.array([0, 1.0]))

    S_exact = np.array([[[ 0.        ,  0.05305165],
                         [ 0.05305165,  0.        ]],
                        [[ 0.05305165,  0.        ],
                         [ 0.        ,  0.05305165]]])
    S_exact = S_exact[:, 0, :]# + S_exact[:, 1, :]

    np.testing.assert_almost_equal(S_exact, S)

def test_hypersingular_vs_deregularized():
    # I differentiated the regularized kernel manually in mathematica
    # and compare with the true hypersingular here.
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    #STUPID BUG!
    angle_one = 0.9
    angle_two = 0.3
    n = np.array([np.cos(angle_one), np.sin(angle_one)])
    m = np.array([np.cos(angle_two), np.sin(angle_two)])
    r = np.array([3.0, 0.5])
    dist = np.sqrt(r[0] * r[0] + r[1] * r[1])
    dr = r / dist
    drdn = dr.dot(n)
    drdm = dr.dot(m)
    k_h = HypersingularKernel(1.0, 0.25)
    const5 = k_h.const5
    S = k_h._call(dist, drdn, drdm, dr, n, m)

    nx = n[0]
    ny = n[1]
    mx = m[0]
    my = m[1]
    rx = dr[0] * dist
    ry = dr[1] * dist
    rs = dist ** 2
    sqrt = np.sqrt
    sxx = (1/rs**2.5)*\
        (const5*
            (mx*
                ((-nx)*rs*(sqrt(rs) + rx**2) +
                 nx*(2*sqrt(rs) + 3*rx**2)*ry**2 +
                 ny*rx*ry*(-2*sqrt(rs) - 4*rs + 3*rx**2 + 6*ry**2)) +
             my*
                (nx*rx*(-2*sqrt(rs) + 2*rs - 3*rx**2)*ry -
                 ny*
                    (rs**1.5 -
                        2*sqrt(rs)*rx**2 -
                        3*rs*rx**2 +
                        3*rx**4 - 2*(rs - 3*rx**2)*ry**2))))
    sxx2 = -(1/rs**2.5)*\
        (const5*(
        (rx**2 - 2*ry**2)*(mx*rx + my*ry)*(nx*rx + ny*ry) +
         mx*nx*sqrt(rs)*rx**2 -
         mx*nx*sqrt(rs)*ry**2 + 2*mx*ny*sqrt(rs)*rx*ry +
         2*my*nx*sqrt(rs)*rx*ry -
         my*ny*sqrt(rs)*rx**2 +
         my*ny*sqrt(rs)*ry**2))
    np.testing.assert_almost_equal(sxx, sxx2)
    syy = -(1/rs**2.5)*\
        (const5*
                (mx*nx*sqrt(rs)*rx**2 -
                 my*ny*sqrt(rs)*rx**2 +
                 2*my*nx*sqrt(rs)*rx*ry +
                 2*mx*ny*sqrt(rs)*rx*ry -
                 mx*nx*sqrt(rs)*ry**2 +
                 my*ny*sqrt(rs)*ry**2 -
                 (mx*rx + my*ry)*(nx*rx + ny*ry)*(2*rx**2 - ry**2)))
    sxy = (3*const5*rx*ry*(mx*rx + my*ry)*(nx*rx + ny*ry))/rs**2.5
    s_deregularized = np.array([[sxx, sxy], [sxy, syy]])
    np.testing.assert_almost_equal(s_deregularized, S)
    # np.testing.assert_almost_equal(syy, S[1, 1])
    # np.testing.assert_almost_equal(sxy, S[0, 1])
    #print syy2
    #print S[1, 1]

def test_hypersingular_vs_regularized():
    from codim1.core.dof_handler import ContinuousDOFHandler
    from codim1.core.mesh import Mesh
    from codim1.core.basis_funcs import BasisFunctions, Solution
    from codim1.core.quad_strategy import QuadStrategy
    from codim1.core.quadrature import QuadGauss
    from codim1.fast.integration import double_integral
    from codim1.fast.mesh import _get_deriv_point, _get_normal

    k_rh = RegularizedHypersingularKernel(1.0, 0.25)
    k_h = HypersingularKernel(1.0, 0.25)
    k_at = AdjointTractionKernel(1.0, 0.25)
    k_t = TractionKernel(1.0, 0.25)
    k_d = DisplacementKernel(1.0, 0.25)

    # mesh = Mesh.simple_line_mesh(3, -1.5, 1.5)
    K = 100
    mesh = Mesh.circular_mesh(K, 1.0)
    bf = BasisFunctions.from_degree(1)
    grad_bf = bf.get_gradient_basis(mesh)
    qs = QuadStrategy(mesh, 10, 10, 10, 10)
    dh = ContinuousDOFHandler(mesh, 1)

    # By the regularization of the hypersingular integral, these two
    # integrations should give the same result.
    el1 = 0
    a = np.zeros((K, 2, 2))
    b = np.zeros((K, 2, 2))
    cr1 = np.zeros(K)
    cr2 = np.zeros(K)
    n1 = np.zeros(K)
    n2x = np.zeros(K)
    n2y = np.zeros(K)
    grad2x = np.zeros(K)
    grad2y = np.zeros(K)
    for i in range(4, K - 3):
        # DETECTIVE WORK!
        el2 = i
        # o_q = QuadGauss(1)
        # i_q = [QuadGauss(1)]
        o_q, i_q = qs.get_quadrature('logr', el1, el2)
        a[i, :, :] = double_integral(mesh, k_rh, grad_bf, grad_bf,
                            o_q, i_q, el2, 0, el1, 0)
        b[i, :, :] = double_integral(mesh, k_h, bf, bf,
                            o_q, i_q, el2, 0, el1, 0)
        cr1[i] = grad_bf.chain_rule(el1, 0.5)[0]
        n1[i] = mesh.get_normal(el1, 0.5)[0]
        n2x[i], n2y[i] = mesh.get_normal(el2, 0.5)
        grad2x[i], grad2y[i] = _get_deriv_point(mesh.basis_fncs.derivs,
                                    mesh.coefficients,
                                    el2,
                                    0.5)
        cr2[i] = -n2x[i] * grad2y[i] + n2y[i] * grad2x[i]
    from matplotlib import pyplot as plt
    # plt.plot(a)
    # plt.plot(b)
    # plt.plot(cr1 / 100.0)
    plt.figure()
    plt.plot(b[:, 0, 0] / a[:, 0, 0])
    plt.plot(b[:, 1, 1] / a[:, 1, 1])
    plt.plot(b[:, 0, 1] / a[:, 0, 1])
    plt.ylim([-1.5, 1.5])
    plt.figure()
    plt.plot(n2y, label='normal')
    plt.plot(grad2y, label='grad')
    plt.plot(cr2, label='cr')
    plt.legend()
    plt.show()
    # c = double_integral(mesh, k_t, bf, bf,
    #                     o_q, i_q, el2, 0, el1, 0)
    # d = double_integral(mesh, k_at, bf, bf,
    #                     o_q, i_q, el2, 0, el1, 0)
    # e = double_integral(mesh, k_d, bf, bf,
    #                     o_q, i_q, el2, 0, el1, 0)
    # print "\n", a, "\n", b, "\n", c, "\n", d, "\n", e
    # np.testing.assert_almost_equal(a, b)
