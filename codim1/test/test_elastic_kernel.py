from codim1.fast.elastic_kernel import *
from codim1.core.dof_handler import ContinuousDOFHandler
from codim1.core.mesh import Mesh
from codim1.core.basis_funcs import BasisFunctions, Solution
from codim1.core.quad_strategy import QuadStrategy
from codim1.core.quadrature import QuadGauss
from codim1.fast.integration import double_integral
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

def test_reverse_normal():
    kernel = TractionKernel(1.0, 0.25)
    a = kernel.call(np.array([1.0, 0.5]),
                    np.zeros(2), np.array([1.0, 0.0]))
    # Only symmetric if we reverse the normal vector too!
    kernel.reverse_normal = True
    b = kernel.call(np.array([-1.0, -0.5]),
                    np.zeros(2), np.array([1.0, 0.0]))
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

def test_hypersingular_vs_regularized():
    # By the regularization of the hypersingular integral, these two
    # integrations should give the same result.
    # I've left
    # LOTS OF DETECTIVE WORK!
    # in this function, because I had a fun (awful?) time figuring out
    # how to get these two integrations to match up... Took three (four?)
    # full days...
    # The integrations are only equal for an interior basis function. If
    # the basis function's support crosses two elements, the point n - 1
    # dimensional term in the integration by parts still influences the
    # result

    k_rh = RegularizedHypersingularKernel(1.0, 0.25)
    k_h = HypersingularKernel(1.0, 0.25)

    K = 30
    mesh = Mesh.circular_mesh(K, 2.0)
    bf = BasisFunctions.from_degree(2)
    grad_bf = bf.get_gradient_basis(mesh)
    qs = QuadStrategy(mesh, 10, 10, 10, 10)
    dh = ContinuousDOFHandler(mesh, 2)

    el1 = 15
    # pp0 = mesh.get_physical_point(el1, 0.5)
    # m = mesh.get_normal(el1, 0.5)
    a = np.zeros((K, 2, 2))
    b = np.zeros((K, 2, 2))
    # qq = np.zeros((K, 2, 2))
    # cr1 = np.zeros(K)
    # cr2 = np.zeros(K)
    # n2x = np.zeros(K)
    # n2y = np.zeros(K)
    # grad2x = np.zeros(K)
    # grad2y = np.zeros(K)
    # k_rh_val = np.zeros((K, 2, 2))
    # k_h_val = np.zeros((K, 2, 2))
    for el2 in range(K):
        if np.abs(el2 - el1) < 2.5:
            continue
        i = 1
        j = 1
        o_q, i_q = qs.get_quadrature('logr', el1, el2)
        a[el2, :, :] = double_integral(mesh, k_rh,
                                             grad_bf, grad_bf,
                                             o_q, i_q, el1, i,
                                             el2, j)
        b[el2, :, :] = double_integral(mesh, k_h, bf, bf,
                            o_q, i_q, el1, i, el2, j)
        # qq[el2, :, :] = double_integral(mesh, k_rh, bf, bf,
        #                     o_q, i_q, el1, 1, el2, 1)
        # # cr1[el2] = grad_bf.chain_rule(el1, 0.5)[0]
        # n2x[el2], n2y[el2] = mesh.get_normal(el2, 0.5)
        # grad2x[el2], grad2y[el2] = _get_deriv_point(mesh.basis_fncs.derivs,
        #                             mesh.coefficients,
        #                             el2,
        #                             0.5)
        # cr2[el2] = -n2x[el2] * grad2y[el2] + n2y[el2] * grad2x[el2]

        # pp = mesh.get_physical_point(el2, 0.5)
        # k_rh_val[el2, :, :] = \
        #         k_rh.call(pp - pp0, m, np.array([n2x[el2], n2y[el2]]))
        # k_h_val[el2, :, :] = \
        #         k_h.call(pp - pp0, m, np.array([n2x[el2], n2y[el2]]))

    # from matplotlib import pyplot as plt
    # plt.plot(a)
    # plt.plot(b)
    # plt.plot(cr1 / 100.0)
    # plt.figure()
    # plt.plot(range(K), a[:, 1, 1, 0, 0], label='axx')
    # plt.plot(range(K), a[:, 0, 0], label='ayy')
    # plt.plot(range(K), a[:, 0, 1], label='axy')
    # plt.plot(range(K), b[:, 1, 1, 0, 0], label='bxx')
    # plt.plot(range(K), b[:, 0, 0], label='byy')
    # plt.plot(range(K), b[:, 0, 1], label='bxy')
    # plt.plot(range(K), qq[:, 0, 0], label='other')
    # plt.plot(grad2x * n2x + grad2y ** 2)
    # plt.legend()

    # plt.figure()
    # plt.plot(a[:, 0, 0] / (1.0 * b[:, 0, 0]))
    # plt.plot(a[:, 1, 1] / (1.0 * b[:, 1, 1]))
    # plt.plot(a[:, 0, 1] / (1.0 * b[:, 0, 1]))
    # plt.ylim([-1.5, 1.5])
    # # plt.plot(grad2x, label='gradx')
    # # plt.plot(grad2y, label='grady')
    # # plt.plot(n2y, label='normal')

    # plt.figure()
    # # plt.plot(k_rh_val[:, 0, 0], label='regularized')
    # # plt.plot(k_h_val[:, 0, 0], label='hyp')
    # plt.plot(k_h_val[:, 0, 0] / k_rh_val[:, 0, 0], label='divided')
    # plt.legend()

    # plt.figure()
    # plt.plot(n2y, label='normal')
    # plt.plot(cr2, label='cr')
    # plt.legend()
    # plt.show()
    np.testing.assert_almost_equal(a, b, 2)

def test_hypersingular_vs_regularized_across_elements():
    # The regularization is only valid for a continuous basis, so the
    # integrations will not be equal unless I account for both elements.
    k_rh = RegularizedHypersingularKernel(1.0, 0.25)
    k_h = HypersingularKernel(1.0, 0.25)

    K = 30
    mesh = Mesh.circular_mesh(K, 2.0)
    bf = BasisFunctions.from_degree(2)
    grad_bf = bf.get_gradient_basis(mesh)
    qs = QuadStrategy(mesh, 10, 10, 10, 10)
    dh = ContinuousDOFHandler(mesh, 2)

    el1a = 15
    el1b = 14
    el2a = 25
    el2b = 26
    o_q, i_q = qs.get_quadrature('logr', el1a, el2a)

    # Four integrals for this matrix term. Two choices of source element
    # and two choices of solution element.
    a1 = double_integral(mesh, k_rh, grad_bf, grad_bf,
                        o_q, i_q, el1a, 0, el2a, 2)
    a2 = double_integral(mesh, k_rh, grad_bf, grad_bf,
                        o_q, i_q, el1a, 0, el2b, 0)
    a3 = double_integral(mesh, k_rh, grad_bf, grad_bf,
                        o_q, i_q, el1b, 2, el2a, 2)
    a4 = double_integral(mesh, k_rh, grad_bf, grad_bf,
                        o_q, i_q, el1b, 2, el2b, 0)
    b1 = double_integral(mesh, k_h, bf, bf,
                        o_q, i_q, el1a, 0, el2a, 2)
    b2 = double_integral(mesh, k_h, bf, bf,
                        o_q, i_q, el1a, 0, el2b, 0)
    b3 = double_integral(mesh, k_h, bf, bf,
                        o_q, i_q, el1b, 2, el2a, 2)
    b4 = double_integral(mesh, k_h, bf, bf,
                        o_q, i_q, el1b, 2, el2b, 0)
    a = a1 + a2 + a3 + a4
    b = b1 + b2 + b3 + b4
    np.testing.assert_almost_equal(a, b)
