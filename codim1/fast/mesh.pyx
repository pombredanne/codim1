# cython: profile=True
import numpy as np
cimport numpy as np
from codim1.fast.basis_funcs cimport evaluate_basis
from libc.math cimport sqrt

cpdef np.ndarray[double, ndim = 1]\
_get_physical_point(np.ndarray[double, ndim = 2] basis_fncs, 
                    np.ndarray[double, ndim = 3] coefficients, 
                    int element_idx,
                    double x_hat):
    """see core/mesh.py for documentation."""
    cdef np.ndarray[double, ndim = 1] phys_pt = np.zeros(2)
    cdef np.ndarray[double, ndim = 1] basis
    cdef int i
    for i in range(basis_fncs.shape[0]):
        basis = evaluate_basis(basis_fncs, i, x_hat)
        phys_pt[0] += coefficients[0, element_idx, i] * basis[0]
        phys_pt[1] += coefficients[1, element_idx, i] * basis[1]
    return phys_pt

cpdef np.ndarray[double, ndim = 1]\
_get_deriv_point(np.ndarray[double, ndim = 2] basis_derivs,
              np.ndarray[double, ndim = 3] coefficients, 
              int element_idx,
              double x_hat):
    cdef np.ndarray[double, ndim = 1] deriv_pt = np.zeros(2)
    cdef np.ndarray[double, ndim = 1] basis
    cdef int i
    for i in range(basis_derivs.shape[0]):
        basis = evaluate_basis(basis_derivs, i, x_hat)
        deriv_pt[0] += coefficients[0, element_idx, i] * basis[0]
        deriv_pt[1] += coefficients[1, element_idx, i] * basis[1]
    return deriv_pt

cpdef double\
_get_jacobian(np.ndarray[double, ndim = 2] basis_derivs,
              np.ndarray[double, ndim = 3] coefficients, 
              int element_idx,
              double x_hat):
    """see core/mesh.py for documentation."""
    cdef np.ndarray[double, ndim = 1] deriv_pt = \
        _get_deriv_point(basis_derivs, coefficients, element_idx, x_hat)
    return sqrt(deriv_pt[0] ** 2 + deriv_pt[1] ** 2)

cpdef np.ndarray[double, ndim = 1] \
_get_normal(np.ndarray[double, ndim = 2] basis_derivs,
            np.ndarray[double, ndim = 3] coefficients, 
            int element_idx,
            double x_hat):
    """see core/mesh.py for documentation."""
    cdef np.ndarray[double, ndim = 1] deriv_pt = \
        _get_deriv_point(basis_derivs, coefficients, element_idx, x_hat)
    cdef double length = sqrt(deriv_pt[0] ** 2 + deriv_pt[1] ** 2)
    return np.array([-deriv_pt[1] / length, deriv_pt[0] / length])
