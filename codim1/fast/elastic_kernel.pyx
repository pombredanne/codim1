# cython: profile=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, log
import math
cdef double pi = math.pi

# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.
# TODO: Lots of replicated code in this file. Try to reduce that.

class ElastostaticKernel:
    """
    A class to contain the elastostatic kernel calculations.
    """
    #cdef double shear_modulus
    #cdef double poisson_ratio
    #cdef double const1, const2, const3, const4
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio

        self.const1 = (1 - 2 * poisson_ratio);
        self.const2 = 1.0 / (4 * pi * (1 - poisson_ratio))
        self.const3 = 1.0 / (8.0 * pi * shear_modulus * (1 - poisson_ratio))
        self.const4 = (3.0 - 4.0 * poisson_ratio)

    def displacement_kernel(self, double rx, double ry, double nx, double ny):
        """
        Return the displacement kernel corresponding to a
        separation of (r[0], r[1])
        This is the effect that is produced $r$ away from a unit point
        force in a full space.
        Goes by many names...
            Green's function
            Kelvin's solution
        """
        cdef double dist_squared = pow(rx, 2) + pow(ry, 2) 
        cdef double dist = sqrt(dist_squared)

        cdef np.ndarray[double, ndim=2] U = np.zeros((2, 2))
        U[0, 0] = -self.const3 * (self.const4 * log(dist) -\
                ((rx ** 2) / dist_squared))
        U[1, 1] = -self.const3 * (self.const4 * log(dist) -\
                ((ry ** 2) / dist_squared))
        U[1, 0] = self.const3 * (rx * ry) / dist_squared
        U[0, 1] = U[1, 0]
        return U

    def traction_kernel(self, double rx, double ry, double nx, double ny):
        """
        Return the traction kernels.
        The normal (n) provided should be the normal to the observation point
        surface, NOT the source point surface.
        """
        # assert(n[0] ** 2 + n[1] ** 2 == 1.0)
        cdef double dist = sqrt(rx ** 2 + ry ** 2)
        cdef double drdn = (rx * nx + ry * ny) / dist
        cdef double dx = rx / dist
        cdef double dy = ry / dist
        cdef np.ndarray[double, ndim=2] T = np.zeros((2, 2))
        cdef double factor = (1 / dist) * self.const2
        T[0, 0] = factor * (-drdn * (self.const1 + 2 * dx ** 2))
        T[1, 1] = factor * (-drdn * (self.const1 + 2 * dy ** 2))
        T[0, 1] = factor * (-drdn * (2 * dx * dy) + \
                self.const1 * (dx * ny - dy * nx))
        T[1, 0] = factor * (-drdn * (2 * dx * dy) + \
                self.const1 * (dy * nx - dx * ny))
        return T

    def hypersingular(self, double rx, double ry, double nx, double ny):
        cdef double dist = sqrt(rx ** 2 + ry ** 2)
        cdef double drdn = (rx * nx + ry * ny) / dist
        cdef np.ndarray[double, ndim=1] dr = np.zeros(2)
        dr[0] = rx / dist
        dr[1] = ry / dist
        cdef np.ndarray[double, ndim=1] n = np.zeros(2)
        n[0] = nx
        n[1] = ny
        cdef np.ndarray[double, ndim=3] S = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    S[i, j, k] = self._hypersingular(i, j, k, dist, drdn,
                                                     dr, n)
        return S

    def _hypersingular(self, int i, int j, int k, 
                       double r, double drdn,
                       np.ndarray[double, ndim=1] dr, 
                       np.ndarray[double, ndim=1] normal):
        Skij = self.shear_modulus / \
                (2 * pi * (1 - self.poisson_ratio)) * 1 / (r ** 2);
        Skij = Skij * ( 2 * drdn * ( self.const1 * (i==j) * dr[k] + \
            self.poisson_ratio * ( dr[j] * (k==i) + dr[i] * (j==k) ) - \
            4 * dr[i] * dr[j] * dr[k] ) + \
            2 * self.poisson_ratio * ( normal[i] * dr[j] * dr[k] + \
            normal[j] * dr[i] * dr[k] ) + \
            self.const1 * \
            ( 2 * normal[k] * dr[i] * dr[j] + normal[j] * (k==i) + \
                normal[i] * (j==k) ) \
            - (1 - 4 * self.poisson_ratio) * normal[k] * (i==j))
        return Skij


    def traction_adjoint(self, double rx, double ry, double nx, double ny):
        cdef double dist = sqrt(rx ** 2 + ry ** 2)
        cdef double drdn = (rx * nx + ry * ny) / dist
        cdef np.ndarray[double, ndim=1] dr = np.zeros(2)
        dr[0] = rx / dist
        dr[1] = ry / dist
        cdef np.ndarray[double, ndim=1] n = np.zeros(2)
        n[0] = nx
        n[1] = ny
        cdef np.ndarray[double, ndim=3] D = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    D[i, j, k] = self._traction_adjoint(i, j, k, dist, drdn,
                                                     dr, n)
        return D

    def _traction_adjoint(self, int i, int j, int k, 
                       double r, double drdn,
                       np.ndarray[double, ndim=1] dr, 
                       np.ndarray[double, ndim=1] normal):
        Dkij = self.const2 * (1 / r) * ( self.const1 * \
                    ( -dr[k] * (i==j) + dr[j] * (k==i) + dr[i] * (j==k) ) + \
                    2 * dr[i] * dr[j] * dr[k] )
        return Dkij
        
