# cython: profile=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, log
import math
cdef double pi = math.pi

# All the boundary kernel functions here are in Frangi and Novati 1996.
# The Dijk and Sijk volume kernels for interior/exterior computation are in 
# the SGBEM book (Sutradhar, Paulio, Gray, 2008).

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
        cdef double drdm = (r[0] * m[0] + r[1] * m[1]) / dist
        cdef double drdn = (r[0] * n[0] + r[1] * n[1]) / dist
        cdef np.ndarray[double, ndim=1] dr = np.zeros(2)
        dr[0] = r[0] / dist
        dr[1] = r[1] / dist
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
        self.const3 = 1.0 / (8.0 * pi * shear_modulus * (1 - poisson_ratio))
        self.const4 = (3.0 - 4.0 * poisson_ratio)

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] Guu = np.zeros((2, 2))
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
        # Unhelpful constant names are unhelpful...
        self.const1 = (1 - 2 * poisson_ratio)
        self.const2 = 1.0 / (4 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] T = np.zeros((2, 2))
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
        # Unhelpful constant names are unhelpful...
        self.const1 = (1 - 2 * poisson_ratio)
        self.const2 = 1.0 / (4 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] T = np.zeros((2, 2))
        cdef double factor = (1 / dist) * self.const2
        T[0, 0] = factor * (drdm * (self.const1 + 2 * dr[0] ** 2))
        T[1, 1] = factor * (drdm * (self.const1 + 2 * dr[1] ** 2))
        T[0, 1] = factor * (drdm * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[0] * m[1] - dr[1] * m[0]))
        T[1, 0] = factor * (drdm * (2 * dr[0] * dr[1]) + \
                self.const1 * (dr[1] * m[0] - dr[0] * m[1]))
        return T

class HypersingularKernel(Kernel):
    """
    Gpp -- 1/r^2 singular in 2D, but integration by parts throws two of the
    (1/r)s onto the basis functions, thus resulting in a log(r) singular
    kernel.
    A derivation of this regularization is given in Frangi, Novati, 1996.
    """
    def __init__(self, double shear_modulus, double poisson_ratio):
        self.singularity_type = 'logr'
        self.const5 = shear_modulus / (2 * pi * (1 - poisson_ratio))

    def _call(self, double dist, double drdn, double drdm,
                    np.ndarray[double, ndim=1] dr,
                    np.ndarray[double, ndim=1] n,
                    np.ndarray[double, ndim=1] m):
        cdef np.ndarray[double, ndim = 2] B = np.zeros((2, 2))
        B[0, 0] = self.const5 * (log(dist) - dr[0] * dr[0])
        B[1, 1] = self.const5 * (log(dist) - dr[1] * dr[1])
        B[1, 0] = self.const5 * (-dr[1] * dr[0])
        B[0, 1] = B[1, 0]
        return B
