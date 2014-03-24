# cython: profile=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, log
import math
cdef double pi = math.pi

# All the kernel functions here are in Frangi and Novati 1996.
############################################################################
# SURFACE KERNELS (Displacement and tractions)
############################################################################

class Kernel(object):
    """
    Base class for kernels.
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio

    def call(self, np.ndarray[double, ndim = 1] r,
                   np.ndarray[double, ndim = 1] m,
                   np.ndarray[double, ndim = 1] n):
        """
        Generic calling method for a elastostatic kernel.
        r is the vector from the source to the solution.
        m is the unit normal vector to the source surface.
        n is the unit normal vector to the solution surface.
        """
        cdef double dist = sqrt(r[0] ** 2 + r[1] ** 2)
        cdef np.ndarray[double, ndim=1] dr = r / dist
        cdef double drdm = dr[0] * m[0] + dr[1] * m[1]
        cdef double drdn = dr[0] * n[0] + dr[1] * n[1]
        return self._call(dist, drdn, drdm, dr, n, m)

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        raise Exception("_call not implemented.")

class DisplacementKernel(Kernel):
    """
    Guu -- log(r) singular in 2D
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.singularity_type = 'logr'
        self.symmetric_matrix = True
        self.const3 = 1.0 / (8.0 * pi * shear_modulus * (1 - poisson_ratio))
        self.const4 = (3.0 - 4.0 * poisson_ratio)

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] Guu = np.empty((2, 2))
        Guu[0, 0] = self.const3 * (-self.const4 * log(dist) +\
                (dr[0] ** 2))
        Guu[1, 1] = self.const3 * (-self.const4 * log(dist) +\
                (dr[1] ** 2))
        Guu[1, 0] = self.const3 * (dr[0] * dr[1])
        Guu[0, 1] = Guu[1, 0]
        return Guu

class TractionKernel(Kernel):
    """
    Gup -- 1/r singular in 2D
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.singularity_type = 'oneoverr'
        self.symmetric_matrix = False
        # Unhelpful constant names are unhelpful...
        self.const1 = (1 - 2 * poisson_ratio)
        self.const2 = 1.0 / (4 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] T = np.empty((2, 2))
        cdef double factor = (1 / dist) * self.const2
        T[0, 0] = factor * (-drdn * (self.const1 + 2 * dr[0] ** 2))
        T[1, 1] = factor * (-drdn * (self.const1 + 2 * dr[1] ** 2))
        T[0, 1] = factor * (-drdn * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[0] * n[1] - dr[1] * n[0]))
        T[1, 0] = factor * (-drdn * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[1] * n[0] - dr[0] * n[1]))
        return T

class AdjointTractionKernel(Kernel):
    """
    Gpu -- 1/r singular in 2D
    Exactly the same code as the standard TractionKernel, just with the
    a different sign and the relevant normal being m instead of n.
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.singularity_type = 'oneoverr'
        self.symmetric_matrix = False
        # Unhelpful constant names are unhelpful...
        self.const1 = (1 - 2 * poisson_ratio)
        self.const2 = 1.0 / (4 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] T = np.empty((2, 2))
        cdef double factor = (1 / dist) * self.const2
        T[0, 0] = factor * (drdm * (self.const1 + 2 * dr[0] ** 2))
        T[1, 1] = factor * (drdm * (self.const1 + 2 * dr[1] ** 2))
        T[0, 1] = factor * (drdm * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[0] * m[1] - dr[1] * m[0]))
        T[1, 0] = factor * (drdm * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[1] * m[0] - dr[0] * m[1]))
        return T

class RegularizedHypersingularKernel(Kernel):
    """
    Gpp -- 1/r^2 singular in 2D, but integration by parts throws two of the
    (1/r)s onto the basis functions, thus resulting in a log(r) singular
    kernel.
    A derivation of this regularization is given in Frangi, Novati, 1996.
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.singularity_type = 'logr'
        self.symmetric_matrix = True
        self.const5 = shear_modulus / (2 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] B = np.empty((2, 2))
        B[0, 0] = self.const5 * (log(dist) - dr[0] * dr[0])
        B[1, 1] = self.const5 * (log(dist) - dr[1] * dr[1])
        B[1, 0] = self.const5 * (-dr[1] * dr[0])
        B[0, 1] = B[1, 0]
        return B

class HypersingularKernel(Kernel):
    """
    The non-regularized hypersingular kernel. Useful for interior 
    computation, though it will be problematic when the interior point
    is very close to the boundary.
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.const1 = (1 - 2 * poisson_ratio)

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] S = np.zeros((2, 2))
        # TODO: Do the math to reduce that ugly expression down to one
        # including the source normal m
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    value = self._hypersingular(i, j, k, dist, 
                                                     dr, drdn, n)
                    S[j, k] += value * m[i]
        return S

    def _hypersingular(self, int i, int j, int k, 
                       double r, np.ndarray[double, ndim=1] dr,
                       double drdn,
                       np.ndarray[double, ndim=1] n):
        """ UGLY!!! """
        cdef double Skij = self.shear_modulus / \
                (2 * pi * (1 - self.poisson_ratio)) * 1 / (r ** 2)
        Skij = Skij * ( 2 * drdn * ( self.const1 * (i==j) * dr[k] + \
            self.poisson_ratio * ( dr[j] * (k==i) + dr[i] * (j==k) ) - \
            4 * dr[i] * dr[j] * dr[k] ) + \
            2 * self.poisson_ratio * ( n[i] * dr[j] * dr[k] + \
            n[j] * dr[i] * dr[k] ) + \
            self.const1 * \
            ( 2 * n[k] * dr[i] * dr[j] + n[j] * (k==i) + \
                n[i] * (j==k) ) \
            - (1 - 4 * self.poisson_ratio) * n[k] * (i==j))
        return Skij
