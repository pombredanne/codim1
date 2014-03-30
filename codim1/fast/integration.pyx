# cython: profile=True
import numpy as np
cimport numpy as np

# TODO: I think the future direction for speeding this up more would be
# to create an "Integrator" class that stores most of the necessary info once
# so that all the python statements in the core of the loop can be removed.
# A c++ implementation could use templates to adapt the integraton
# to matrix entries vs. rhs values, linear elements, high order, etc.
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

    # Just store some variables in a typed way to speed things up 
    # inside the loop
    cdef bint is_linear = mesh.is_linear

    # Jacobian determinants are necessary to scale the integral with the
    # change of variables. 
    cdef double src_jacobian = 1.0
    cdef double soln_jacobian = 1.0

    # The l_normal is needed for the traction kernel -- the solution normal.
    # The k_normal is normally needed for the adjoint traction kernel
    # and for the hypersingular kernel -- the source normal.
    cdef np.ndarray[double, ndim = 1] k_normal, l_normal

    # If the mesh is linear then the jacobian and normal vectors will be the
    # same at all quadrature points, so just grab them here once.
    if is_linear:
        src_jacobian = np.array(mesh.get_jacobian(k, 0.0))
        soln_jacobian = np.array(mesh.get_jacobian(l, 0.0))
        k_normal = np.array(mesh.get_normal(k, 0.0))
        l_normal = np.array(mesh.get_normal(l, 0.0))

    cdef int q_src_pt_index, q_soln_pt_index
    cdef double q_pt_src, w_src, q_pt_soln, w_soln
    cdef np.ndarray[double, ndim = 1] src_basis_fnc, soln_basis_fnc
    cdef np.ndarray[double, ndim = 1] phys_src_pt, phys_soln_pt
    cdef np.ndarray[double, ndim = 1] r = np.empty(2)
    cdef np.ndarray[double, ndim = 2] k_val
    cdef np.ndarray[double, ndim = 1] q_pts_soln, q_w_soln
    cdef int idx_x, idx_y

    # The outer quadrature uses a standard nonsingular quadrature formula
    cdef np.ndarray[double, ndim = 1] q_pts = src_quadrature.x
    cdef np.ndarray[double, ndim = 1] w = src_quadrature.w
    for q_src_pt_index in range(q_pts.shape[0]):
        q_pt_src = q_pts[q_src_pt_index]
        w_src = w[q_src_pt_index]

        # Translate from reference segment coordinates to 
        # real, physical coordinates
        phys_src_pt = np.array(mesh.get_physical_point(k, q_pt_src))

        if not is_linear:
            # Get the jacobian at this quad point
            # The basis derivative chain rule at this point.
            src_jacobian = np.array(mesh.get_jacobian(k, q_pt_src))
            # Get the normal to this element at this quadrature point
            k_normal = np.array(mesh.get_normal(k, q_pt_src))


        # The basis functions should be evaluated on reference
        # coordinates
        src_basis_fnc = src_basis_fncs.evaluate(k, i, 
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

            phys_soln_pt = np.array(mesh.get_physical_point(l, q_pt_soln))

            if not is_linear:
                soln_jacobian = np.array(mesh.get_jacobian(l, q_pt_soln))
                l_normal = np.array(mesh.get_normal(l, q_pt_soln))



            soln_basis_fnc = soln_basis_fncs.evaluate(l, j, 
                                    q_pt_soln, phys_soln_pt)

            # Separation of the two quadrature points, use real,
            # physical coordinates!
            # From source to solution.
            r[0] = phys_soln_pt[0] - phys_src_pt[0]
            r[1] = phys_soln_pt[1] - phys_src_pt[1]

            # Actually evaluate the kernel.
            k_val = np.array(kernel.call(r, k_normal, l_normal))

            # Account for the vector form of the problem.
            # and weight by the quadrature values and the jacobian
            for idx_x in range(2):
                for idx_y in range(2):
                    result[idx_x, idx_y] += k_val[idx_x, idx_y] * \
                            src_basis_fnc[idx_x] * \
                            soln_basis_fnc[idx_y] * \
                            src_jacobian * soln_jacobian * \
                            w_soln * w_src
    return result

def single_integral(mesh, kernel, src_basis_fncs, soln_basis_fncs,
                    quadrature, int k, int i, int j):
    """
    Performs a single integral over one element. The operations are all 
    almost identical to those in the double_integral method. Thus, read 
    through that method for details. 

    A key difference with single_integral is that the kernel function here
    is expected to just be a standard function with a location 
    parameter. The double integral function takes a kernel class object with 
    a call method taking a separation input. K(x) vs. K(x - y)
    """
    cdef np.ndarray[double, ndim = 2] result = np.zeros((2, 2))

    cdef bint is_linear = mesh.is_linear

    cdef double jacobian    
    cdef np.ndarray[double, ndim = 1] k_normal
    cdef np.ndarray[double, ndim = 1] phys_pt
    cdef np.ndarray[double, ndim = 1] src_basis_fnc, soln_basis_fnc
    cdef np.ndarray[double, ndim = 2] k_val
    cdef int idx_x, idx_y

    if is_linear:
        jacobian = np.array(mesh.get_jacobian(k, 0.0))
        k_normal = np.array(mesh.get_normal(k, 0.0))

    cdef np.ndarray[double, ndim = 1] q_pts = quadrature.x
    cdef np.ndarray[double, ndim = 1] wts = quadrature.w
    for (q_pt, w) in zip(q_pts, wts):
        phys_pt = np.array(mesh.get_physical_point(k, q_pt))

        if not is_linear:
            jacobian = np.array(mesh.get_jacobian(k, q_pt))
            k_normal = np.array(mesh.get_normal(k, q_pt))

        src_basis_fnc = src_basis_fncs.evaluate(k, i, q_pt, phys_pt)
        soln_basis_fnc = soln_basis_fncs.evaluate(k, j, q_pt, phys_pt)

        k_val = np.array(kernel(phys_pt, k_normal))

        for idx_x in range(2):
            for idx_y in range(2):
                result[idx_x, idx_y] += k_val[idx_x, idx_y] * \
                        src_basis_fnc[idx_x] * soln_basis_fnc[idx_y] * \
                        jacobian * w
    return result
