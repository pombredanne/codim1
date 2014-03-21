import numpy as np
cimport numpy as np
cpdef np.ndarray[double, ndim = 1]\
_get_physical_point(np.ndarray[double, ndim = 2] basis_fncs, 
                    np.ndarray[double, ndim = 3] coefficients, 
                    int element_idx,
                    double x_hat)

cpdef double\
_get_jacobian(np.ndarray[double, ndim = 2] basis_fncs,
              np.ndarray[double, ndim = 3] coefficients, 
              int element_idx,
              double x_hat)

cpdef np.ndarray[double, ndim = 1] \
_get_normal(np.ndarray[double, ndim = 2] basis_fncs,
            np.ndarray[double, ndim = 3] coefficients, 
            int element_idx,
            double x_hat)
