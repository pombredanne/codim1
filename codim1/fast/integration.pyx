# cython: profile=True
import numpy as np
cimport numpy as np
from get_physical_points cimport get_physical_points
from basis_funcs cimport evaluate_basis

# TODO: I think the future direction for speeding this up more would be
# to create an "Integrator" class that stores most of the necessary info once
# so that all the python statements in the core of the loop can be removed.
# A true c++ implementation could use templates to adapt the integraton
# to matrix elements vs. rhs element, etc.
# TODO: I think this function has reversed the order of integration from the
# standard. Flip it back to standard. Shouldn't matter, because the domains 
# are independently defined, but conforming to standards is generally good.
def double_integral(mesh, kernel, 
                    src_basis_fncs, soln_basis_fncs, 
                    src_quadrature, soln_quadrature, 
                    int k, int i, int l, int j):
    """
    Performs a double integral over a pair of elements with the
    provided quadrature rule.

    Can handle matrix elements or elements of a RHS vector.

    In a sense, this is the core method of any BEM implementation.
    """
    cdef np.ndarray[double, ndim = 2] result = np.zeros((2, 2))

    # Jacobian determinants are necessary to scale the integral with the
    # change of variables.
    cdef double src_jacobian = mesh.get_element_jacobian(k)
    cdef double soln_jacobian = mesh.get_element_jacobian(l)

    # The l_normal is needed for the traction kernel -- the solution normal.
    # The k_normal is normally needed for the adjoint traction kernel
    # and for the hypersingular kernel -- the source normal.
    cdef np.ndarray[double, ndim = 1] k_normal = mesh.normals[k]
    cdef np.ndarray[double, ndim = 1] l_normal = mesh.normals[l]

    # Just store some variables in a typed way to speed things up 
    # inside the loop
    cdef np.ndarray[long, ndim = 2] element_to_vertex = mesh.element_to_vertex
    cdef np.ndarray[double, ndim = 2] vertices = mesh.vertices

    # The outer quadrature uses a standard nonsingular quadrature formula
    cdef np.ndarray[double, ndim = 1] q_pts = src_quadrature.x
    cdef np.ndarray[double, ndim = 1] w = src_quadrature.w
    cdef int q_src_pt_index, q_soln_pt_index
    cdef double q_pt_src, w_src, q_pt_soln, w_soln
    cdef np.ndarray[double, ndim = 1] src_basis_fnc
    cdef np.ndarray[double, ndim = 1] soln_basis_fnc
    cdef np.ndarray[double, ndim = 1] phys_src_pt, phys_soln_pt
    cdef np.ndarray[double, ndim = 1] r
    cdef np.ndarray[double, ndim = 2] k_val
    for q_src_pt_index in range(q_pts.shape[0]):
        q_pt_src = q_pts[q_src_pt_index]
        w_src = w[q_src_pt_index]

        # Translate from reference segment coordinates to 
        # real, physical coordinates
        phys_src_pt = get_physical_points(element_to_vertex, vertices,
                                          k, q_pt_src)

        # The basis functions should be evaluated on reference
        # coordinates
        src_basis_fnc = src_basis_fncs.evaluate_basis(i, 
                                q_pt_src, phys_src_pt)

        # If the integrand is singular, we need to use the appropriate
        # inner quadrature method. Which points the inner quadrature
        # chooses will depend on the current outer quadrature point
        # which will be the point of singularity, assuming same element
        q_pts_soln = soln_quadrature[q_src_pt_index].x
        q_w_soln = soln_quadrature[q_src_pt_index].w

        for q_soln_pt_index in range(q_pts_soln.shape[0]):
            q_pt_soln = q_pts_soln[q_soln_pt_index]
            w_soln = q_w_soln[q_soln_pt_index]

            phys_soln_pt = get_physical_points(element_to_vertex,
                                               vertices, l, q_pt_soln)

            soln_basis_fnc = soln_basis_fncs.evaluate_basis(j, 
                                    q_pt_soln, phys_soln_pt)

            # Separation of the two quadrature points, use real,
            # physical coordinates!
            # From source to solution.
            r = phys_soln_pt - phys_src_pt

            # Actually evaluate the kernel.
            k_val = kernel.call(r, k_normal, l_normal)

            # Weight by the quadrature values
            k_val *= w_soln * w_src

            # Account for the vector form of the problem.
            for idx_x in range(2):
                for idx_y in range(2):
                    result[idx_x, idx_y] += k_val[idx_x, idx_y] * \
                            src_basis_fnc[idx_x] * soln_basis_fnc[idx_y]

    result *= src_jacobian * soln_jacobian
    return result
