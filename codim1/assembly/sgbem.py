import numpy as np
from codim1.fast_lib import double_integral, single_integral,\
    MassMatrixKernel, ZeroBasis, ConstantBasis
from itertools import product
from functools import partial

"""
These functions traverse the mesh and form the symmetric galerkin
matrices and right hand side vectors.

The implementation here follows equations 14 through 17 in the Bonnet 98
paper very closely.

This assumes that all the boundary conditions have already been attached to
the relevant element in the mesh.
"""
def sgbem_assemble(mesh, kernel_set):
    # Form the empty linear system
    total_dofs = mesh.total_dofs
    lhs_matrix = np.zeros((total_dofs, total_dofs))
    rhs_matrix = np.zeros((total_dofs, total_dofs))
    mass_matrix = np.zeros((total_dofs, total_dofs))

    # Set the kernels for each type of boundary condition.
    which_kernels = _make_which_kernels(kernel_set)

    # Traverse the mesh and assemble the relevant terms
    for e_k in mesh:
        # Add the mass matrix term to the right hand side.
        _element_mass_rhs(mass_matrix, e_k)
        for e_l in mesh:
            # Compute and add the RHS and matrix terms to the system.
            _element_pair(lhs_matrix, e_k, e_l, which_kernels, "matrix")
            _element_pair(rhs_matrix, e_k, e_l, which_kernels, "rhs")


    # Combine the two rhs terms
    rhs = np.sum(rhs_matrix, axis = 1)
    mass_rhs = np.sum(mass_matrix, axis = 1)
    rhs += 0.5 * mass_rhs

    # Return the fully assembled linear system
    return lhs_matrix, rhs

def _element_mass_rhs(matrix, e_k):
    # Because the term is identical (just replace u by t) for both
    # integral equations, this function does not care about the BC type
    bc_basis = e_k.bc.basis
    q_info = e_k.qs.get_nonsingular_minpts()
    kernel = MassMatrixKernel(0, 0)
    for i in range(e_k.basis.n_fncs):
        for j in range(bc_basis.n_fncs):
            M_local = single_integral(e_k.mapping.eval,
                              kernel,
                              e_k.basis,
                              bc_basis,
                              q_info,
                              i, j)
            matrix[e_k.dofs[0, i], e_k.dofs[0, j]] += M_local[0][0]
            matrix[e_k.dofs[1, i], e_k.dofs[1, j]] += M_local[1][1]

def _choose_basis(basis, is_gradient):
    if is_gradient:
        pt_src_info = zip(basis.point_sources, basis.point_source_dependency)
        return basis.get_gradient_basis(), pt_src_info
    return basis, []

def _element_pair(matrix, e_k, e_l, which_kernels, rhs_or_matrix):
    # Determine whether to use the boundary condition or the solution basis
    # as the inner basis function
    if rhs_or_matrix == "rhs":
        init_e_l_basis = e_l.bc.basis
    else:
        init_e_l_basis = e_l.basis

    # If either of the bases are the zero basis, then don't compute
    # anything
    if type(e_k.basis) is ZeroBasis or \
       type(init_e_l_basis) is ZeroBasis:
        return

    # Determine which kernel and which bases to use
    kernel, factor = which_kernels[e_k.bc.type][e_l.bc.type][rhs_or_matrix]
    if kernel is None:
        return

    # Decide whether to use the basis or its gradient
    e_k_basis, e_k_pt_srcs = _choose_basis(e_k.basis, kernel.test_gradient)
    e_l_basis, e_l_pt_srcs = _choose_basis(init_e_l_basis, kernel.soln_gradient)

    # Now that we might have taken a derivative, check for ZeroBases again.
    if type(e_k_basis) is ZeroBasis or \
       type(e_l_basis) is ZeroBasis:
        return

    # Determine what quadrature formula to use
    quad_outer, quad_inner = e_k.qs.get_quadrature(
                            kernel.singularity_type, e_k, e_l)

    # Loop over basis function pairs and integrate!
    for i in range(e_k.basis.n_fncs):
        for j in range(e_l.basis.n_fncs):
            integral = \
                double_integral(e_k.mapping.eval, e_l.mapping.eval,
                                kernel, e_k_basis, e_l_basis,
                                quad_outer, quad_inner, i, j)

            # Add the integrated term in the appropriate location
            for idx1 in range(2):
                for idx2 in range(2):
                    matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, j]] +=\
                            factor * integral[idx1][idx2]

    # Filter out the point sources that we can safely ignore
    # many of these will be ignored because there will be an equal and
    # opposite point source contribution from the adjacent element.
    # TODO!
    # Currently, I ignore all point source on the test function side
    # of the problem, because these should be handled by the continuity
    # of the displacement field. I should probably think about this
    # a bit more...
    e_k_pt_srcs = []

    # This probably explains some of the problems I was having with the
    # constant traction crack problem.

    # I also ignore all point source if we are dealing
    if rhs_or_matrix == "matrix":
        return

    # Loop over point sources and integrate!
    # All cross multiplications are necessary.
    # the pt sources are tuples like ((node, str_x, str_y), local_dof)
    # for e_k_pt in e_k_pt_srcs:
    #     phys_pt_k = e_k.mapping.get_physical_point(e_k_pt[0][0])
    #     kernel.set_interior_data(phys_pt_k,
    #     for j in range(e_l.basis.n_fncs):
    #         pass
    #     for e_l_pt in e_k_pt_srcs:
    #         phys_pt_l = e_l.mapping.get_physical_point(e_l_pt[0][0])
    for i in range(e_k.basis.n_fncs):
        for e_l_pt in e_l_pt_srcs:
            e_l_dof = e_l_pt[1]
            phys_pt_l = e_l.mapping.get_physical_point(e_l_pt[0][0])
            normal_l = e_l.mapping.get_normal(e_l_pt[0][0])
            strength = ConstantBasis([e_l_pt[0][1], e_l_pt[0][2]])
            kernel.set_interior_data(phys_pt_l, normal_l)
            integral = single_integral(e_k.mapping.eval,
                                   kernel,
                                   e_k_basis,
                                   strength,
                                   quad_inner[e_l_dof],
                                   i, 0)
            for idx1 in range(2):
                for idx2 in range(2):
                    matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, e_l_dof]] += \
                        factor * integral[idx1][idx2]

def _make_which_kernels(kernel_set):
    """
    A table indicating which kernel should be used for the matrix term
    and which kernel should be used for the RHS term given the type of
    boundary condition on each of the elements under consideration.

    The outer boundary condition type is the BC for the test function and
    the inner boundary condition type is the BC for the solution element.

    Use this like:
    which_kernels[e_k.bc.type][e_l.bc.type]["matrix"]
    or
    which_kernels[e_k.bc.type][e_l.bc.type]["rhs"]
    """
    which_kernels = \
        {
            "displacement":
            {
                "displacement":
                {
                    "matrix": (kernel_set.k_d, 1),
                    "rhs": (kernel_set.k_t, 1)
                },
                "traction":
                {
                    "matrix": (kernel_set.k_t, -1),
                    "rhs": (kernel_set.k_d, -1)
                }
            },
            "traction":
            {
                "displacement":
                {
                    "matrix": (kernel_set.k_tp, 1),
                    "rhs": (kernel_set.k_rh, 1)
                },
                "traction":
                {
                    "matrix": (kernel_set.k_rh, -1),
                    "rhs": (kernel_set.k_tp, -1)
                }
            },
            "crack_traction":
            {
                "crack_traction":
                {
                    "matrix": (kernel_set.k_rh, -0.5),
                    "rhs": (None, 0)
                }
                # "displacement":
                # {
                #     "matrix": (kernel_set.k_tp, 1),
                #     "rhs": (kernel_set.k_rh, 1)
                # },
                # "traction":
                # {
                #     "matrix": (kernel_set.k_rh, -1),
                #     "rhs": (kernel_set.k_tp, -1)
                # }
            }
        }
    return which_kernels
