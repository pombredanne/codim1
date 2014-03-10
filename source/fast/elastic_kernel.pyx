# cython: profile=True
import numpy as np

class ElastostaticKernel:
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

    def displacement_kernel(self, double rx, double ry,
                                  double nx, double ny):
        """
        Return the displacement kernel corresponding to a
        separation of (r[0], r[1])
        This is the effect that is produced $r$ away from a unit point
        force in a full space.
        Goes by many names...
            Green's function
            Kelvin's solution
        """
        cdef double const3 = self.const3
        cdef double const4 = self.const4
        cdef double dist_squared = r[0] ** 2 + r[1] ** 2
        cdef double dist = np.sqrt(dist_squared)

        cdef double[:, :] U = np.zeros((2, 2))
        U[0, 0] = -self.const3 * (self.const4 * np.log(dist) -\
                ((r[0] ** 2) / dist_squared))
        U[1, 1] = -self.const3 * (self.const4 * np.log(dist) -\
                ((r[1] ** 2) / dist_squared))
        U[1, 0] = self.const3 * (r[0] * r[1]) / dist_squared
        U[0, 1] = U[1, 0]
        return U

    def traction_kernel(self, r, n):
        """
        Return the traction kernels.
        The normal (n) provided should be the normal to the observation point
        surface, NOT the source point surface.
        """
        # assert(n[0] ** 2 + n[1] ** 2 == 1.0)
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
