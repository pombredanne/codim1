# cython: profile=True
import numpy as np
cimport numpy as np

cpdef evaluate_basis(np.ndarray[double, ndim = 2] basis, long i, double x):
    """
        Evaluates the i-th lagrange polynomial at x.
    """
    cdef double retval = 0.0
    cdef int c_idx = 0
    cdef int order = basis.shape[0]
    for c_idx in range(order):
        retval += basis[i][c_idx] * \
                  x ** (order - 1 - c_idx)
    return retval
