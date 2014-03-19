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
    kernel = HypersingularKernel(1.0, 0.25)
    W = kernel.call(np.array([2.0, 0.0]),
                    np.array([0, 1.0]),
                    np.array([0, 0.0]))
    W_exact = np.array([[2 * (np.log(2) + 1) / (3 * np.pi), 0],
                        [0, 2 * np.log(2) / (3 * np.pi)]])
