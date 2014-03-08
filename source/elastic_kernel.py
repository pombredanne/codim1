import numpy as np

class ElastostaticKernel(object):
    """
    A class to contain the elastostatic kernel calculations.
    """
    def __init__(self, shear_modulus, poisson_ratio):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio

        self.const1 = (1 - 2 * poisson_ratio);
        self.const2 = 1.0 / (4 * np.pi * (1 - poisson_ratio))
        self.const3 = 1.0 / (8.0 * np.pi * shear_modulus * (1 - poisson_ratio))
        self.const4 = (3.0 - 4.0 * poisson_ratio)

    def displacement_singular(self, r, n):
        """The singular (log(r)) part of the displacement kernel"""
        dist = np.sqrt(r[0] ** 2 + r[1] ** 2)
        U = np.zeros((2, 2))
        U[0, 0] = -self.const3 * self.const4 * np.log(dist)
        U[1, 1] = -self.const3 * self.const4 * np.log(dist)
        return U

    def displacement_nonsingular(self, r, n):
        """The nonsingular part of the displacement kernel"""
        # These kernels are well behaved at r = 0, but the computer will still
        # scream if you divide by zero. So, we just return zero.
        dist_squared = r[0] ** 2 + r[1] ** 2
        if dist_squared == 0:
            return np.ones((2, 2))
        dist = np.sqrt(dist_squared)
        U = np.zeros((2, 2))
        U[0, 0] = self.const3 * ((r[0] ** 2) / dist_squared)
        U[1, 1] = self.const3 * ((r[1] ** 2) / dist_squared)
        U[1, 0] = self.const3 * (r[0] * r[1]) / dist_squared
        U[0, 1] = U[1, 0]
        return U

    def displacement_kernel(self, r, n):
        """
        Return the displacement kernel corresponding to a
        separation of (r[0], r[1])
        This is the effect that is produced $r$ away from a unit point
        force in a full space.
        Goes by many names...
            Green's function
            Kelvin's solution
        """
        return self.displacement_singular(r, n) \
            + self.displacement_nonsingular(r, n)

    def traction_kernel(self, r, n):
        """
        Return the traction kernels.
        The normal (n) provided should be the normal to the observation point
        surface, NOT the source point surface.
        """
        assert(n[0] ** 2 + n[1] ** 2 == 1.0)
        dist = np.sqrt(r[0] ** 2 + r[1] ** 2)
        drdn = r.dot(n) / dist
        dr = np.array([r[0] / dist, r[1] / dist])
        T = np.zeros((2, 2))
        T[0, 0] = self.traction_kernel_element(0, 0, dist, dr, drdn, n)
        T[1, 1] = self.traction_kernel_element(1, 1, dist, dr, drdn, n)
        T[0, 1] = self.traction_kernel_element(0, 1, dist, dr, drdn, n)
        T[1, 0] = -T[0, 1]
        return T

    def traction_kernel_element(self, i, j, dist, dr, drdn, normal):
        return (1 / dist) * self.const2 *\
                (-drdn * (self.const1 * (i == j) +  2 * (dr[i] * dr[j])) + \
                 self.const1 * (dr[i] * normal[j] - dr[j] * normal[i]))

################################################################################
# TESTS                                                                        #
################################################################################

def test_traction_kernel_elements():
    E = 1e5
    nu = 0.3
    shear_modulus = E / (2 * (1 + nu))
    kernel = ElastostaticKernel(shear_modulus, nu)
    r = 4.7285
    dr = np.array([0, 1])
    drdn = 0
    normal = np.array([-1, 0])
    T = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
           T[i, j] = kernel.traction_kernel_element(i, j, r, dr, drdn, normal)
    exact = np.array([[0, 0.0096],[-0.0096, 0]])
    np.testing.assert_almost_equal(exact, T, 4)

def test_displacement_nonsingular_for_r_equal_to_0():
    kernel = ElastostaticKernel(30e9, 0.25)
    u_nonsing = kernel.displacement_nonsingular(np.array([0.0, 0.0]), 0.0)
    assert(not np.isnan(np.sum(u_nonsing)))

def test_displacement_symmetry():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.displacement_kernel([1.0, 0.5], 0.0)
    np.testing.assert_almost_equal(a - a.T, np.zeros_like(a))

def test_displacement_mirror_symmetry():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.displacement_kernel(np.array([1.0, 0.5]), 0.0)
    b = kernel.displacement_kernel(np.array([-1.0, -0.5]), 0.0)
    np.testing.assert_almost_equal(a, b)

def test_traction_mirror_symmety():
    kernel = ElastostaticKernel(1.0, 0.25)
    a = kernel.traction_kernel(np.array([1.0, 0.5]), np.array([1.0, 0.0]))
    # Only symmetric if we reverse the normal vector too!
    b = kernel.traction_kernel(np.array([-1.0, -0.5]), np.array([-1.0, 0.0]))
    np.testing.assert_almost_equal(a, b)

def test_displacement():
    kernel = ElastostaticKernel(1.0, 0.25)
    G = kernel.displacement_kernel((2.0, 0.0), 0.0)
    np.testing.assert_almost_equal(G[0, 0],
                                    (2 * np.log(1 / 2.0) + 1) / (6 * np.pi))
    np.testing.assert_almost_equal(G[1, 0], 0.0)
    np.testing.assert_almost_equal(G[0, 1], 0.0)
    np.testing.assert_almost_equal(G[1, 1],
                                    (2 * np.log(1 / 2.0)) / (6 * np.pi))

def test_traction():
    kernel = ElastostaticKernel(1.0, 0.25)
    H = kernel.traction_kernel(np.array((2.0, 0.0)), (0, 1.0))
    np.testing.assert_almost_equal(H[0, 1],
                                    1 / (6 * np.pi * 2.0))
    np.testing.assert_almost_equal(H[0, 0], 0.0)
    np.testing.assert_almost_equal(H[1, 1], 0.0)
    np.testing.assert_almost_equal(H[1, 0], -H[0, 1])

