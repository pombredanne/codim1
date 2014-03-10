import numpy as np
cimport numpy as np
cimport cython

@cython.profile(True)
cpdef get_physical_points(np.ndarray[long, ndim = 2] element_to_vertex,
                          np.ndarray[double, ndim = 2] vertices,
                          int element_id, double reference_pt):
    cdef int vertex1 = element_to_vertex[element_id, 0]
    cdef int vertex2 = element_to_vertex[element_id, 1]
    cdef double pt2_minus_pt1, pt1
    cdef np.ndarray[double, ndim = 1] physical_pts = np.empty(2)
    cdef int d
    for d in range(2):
        pt1 = vertices[vertex1][d]
        pt2_minus_pt1 = vertices[vertex2][d] - pt1
        physical_pts[d] = pt1 + reference_pt * pt2_minus_pt1
    return physical_pts

