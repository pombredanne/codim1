# cython: profile=True
import numpy as np
cimport numpy as np
from get_physical_points cimport get_physical_points
from basis_funcs cimport evaluate_basis

def double_integral(mesh, basis_funcs, kernel, 
                    outer_quadrature, inner_quadrature, 
                    int k, int i, int l, int j):
    """
    Performs a double integral over a pair of elements with the
    provided quadrature rule.

    In a sense, this is the core method of any BEM implementation.

    Warning: This function modifies the "result" input.
    """
    cdef np.ndarray[double, ndim = 2] result = np.zeros((2, 2))

    # Jacobian determinants are necessary to scale the integral with the
    # change of variables.
    cdef double src_jacobian = mesh.get_element_jacobian(k)
    cdef double soln_jacobian = mesh.get_element_jacobian(l)

    # The normal is the one on the soln integration element.
    # This is clear if you remember the source is actually a point
    # and thus has no defined normal. We are integrating over many point
    # sources.
    cdef np.ndarray[double, ndim = 1] normal = mesh.normals[l]

    # Just store some variables in a typed way to speed things up 
    # inside the loop
    cdef np.ndarray[long, ndim = 2] element_to_vertex = mesh.element_to_vertex
    cdef np.ndarray[double, ndim = 2] vertices = mesh.vertices
    cdef np.ndarray[double, ndim = 2] basis = basis_funcs.fncs

    # The outer quadrature uses a standard nonsingular quadrature formula
    cdef np.ndarray[double, ndim = 1] q_pts = outer_quadrature.x
    cdef np.ndarray[double, ndim = 1] w = outer_quadrature.w
    cdef int q_src_pt_index, q_soln_pt_index
    cdef double q_pt_src, w_src, q_pt_soln, w_soln
    cdef double src_basis_fnc, soln_basis_fnc
    cdef np.ndarray[double, ndim = 1] phys_src_pt, phys_soln_pt
    cdef np.ndarray[double, ndim = 1] r
    cdef np.ndarray[double, ndim = 2] k_val
    for q_src_pt_index in range(q_pts.shape[0]):
        q_pt_src = q_pts[q_src_pt_index]
        w_src = w[q_src_pt_index]

        phys_src_pt = get_physical_points(element_to_vertex, vertices,
                                          k, q_pt_src)
        # The basis functions should be evaluated on reference
        # coordinates
        src_basis_fnc = evaluate_basis(basis, i, q_pt_src)

        # If the integrand is singular, we need to use the appropriate
        # inner quadrature method. Which points the inner quadrature
        # chooses will depend on the current outer quadrature point
        # which will be the point of singularity, assuming same element
        q_pts_soln = inner_quadrature[q_src_pt_index].x
        q_w_soln = inner_quadrature[q_src_pt_index].w

        for q_soln_pt_index in range(q_pts_soln.shape[0]):
            q_pt_soln = q_pts_soln[q_soln_pt_index]
            w_soln = q_w_soln[q_soln_pt_index]

            soln_basis_fnc = evaluate_basis(basis, j, q_pt_soln)

            # Separation of the two quadrature points, use real,
            # physical coordinates!
            phys_soln_pt = get_physical_points(element_to_vertex,
                                               vertices, l, q_pt_soln)

            # From source to solution.
            r = phys_soln_pt - phys_src_pt

            # Actually evaluate the kernel.
            k_val = kernel(r[0], r[1], normal[0], normal[1])

            # Actually perform the quadrature
            result += k_val * src_basis_fnc * soln_basis_fnc *\
                      src_jacobian * soln_jacobian *\
                      w_soln * w_src
    return result
