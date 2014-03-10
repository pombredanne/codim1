from fast.elastic_kernel import ElastostaticKernel
import numpy as np

def test_traction_kernel_elements():
    E = 1e5
    nu = 0.3
    shear_modulus = E / (2 * (1 + nu))
    kernel = ElastostaticKernel(shear_modulus, nu)
    T = kernel.traction_kernel(0, 4.7285, -1, 0)
    exact = np.array([[0, 0.0096],[-0.0096, 0]])
    np.testing.assert_almost_equal(exact, T, 4)

def test_displacement_symmetry():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.displacement_kernel(1.0, 0.5, 0.0, 0.0)
    np.testing.assert_almost_equal(a - a.T, np.zeros_like(a))

def test_displacement_mirror_symmetry():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.displacement_kernel(1.0, 0.5, 0.0, 0.0)
    b = kernel.displacement_kernel(-1.0, -0.5, 0.0, 0.0)
    np.testing.assert_almost_equal(a, b)

def test_traction_mirror_symmety():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.traction_kernel(1.0, 0.5, 1.0, 0.0)
    # Only symmetric if we reverse the normal vector too!
    b = kernel.traction_kernel(-1.0, -0.5, -1.0, 0.0)
    np.testing.assert_almost_equal(a, b)

def test_displacement():
    kernel = ElastostaticKernel(1.0, 0.25)
    G = kernel.displacement_kernel(2.0, 0.0, 0.0, 0.0)
    np.testing.assert_almost_equal(G[0, 0],
                                    (2 * np.log(1 / 2.0) + 1) / (6 * np.pi))
    np.testing.assert_almost_equal(G[1, 0], 0.0)
    np.testing.assert_almost_equal(G[0, 1], 0.0)
    np.testing.assert_almost_equal(G[1, 1],
                                    (2 * np.log(1 / 2.0)) / (6 * np.pi))

def test_traction():
    kernel = ElastostaticKernel(1.0, 0.25)
    H = kernel.traction_kernel(2.0, 0.0, 0, 1.0)
    np.testing.assert_almost_equal(H[0, 1],
                                    1 / (6 * np.pi * 2.0))
    np.testing.assert_almost_equal(H[0, 0], 0.0)
    np.testing.assert_almost_equal(H[1, 1], 0.0)
    np.testing.assert_almost_equal(H[1, 0], -H[0, 1])

