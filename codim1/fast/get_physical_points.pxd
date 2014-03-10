cimport numpy as np
cpdef get_physical_points(np.ndarray[long, ndim = 2] element_to_vertex,
                          np.ndarray[double, ndim = 2] vertices,
                          int element_id, double reference_pt)
