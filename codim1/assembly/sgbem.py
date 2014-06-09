from codim1.fast_lib import double_integral, single_integral,\
    MassMatrixKernel, ZeroBasis, ConstantBasis
from which_kernels import _make_which_kernels
from shared import _choose_basis
from codim1.post.interior import sgbem_interior

import numpy as np

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
        if e_k.bc.type == "crack_displacement":
            # # If displacement discontinuity, we just want the identity
            # # matrix to remove the outer un-integrable integral
            _identity_matrix(lhs_matrix, e_k)
            for i in range(e_k.basis.n_fncs):
                ref_pt = e_k.basis.nodes[i]
                normal = e_k.mapping.get_normal(ref_pt)
                phys_pt = e_k.mapping.get_physical_point(ref_pt)
                interior_val = sgbem_interior(mesh, phys_pt,
                                normal, kernel_set, "crack_traction")
                if not np.isnan(interior_val).any():
                    rhs_matrix[e_k.dofs[0, i], 0] = interior_val[0]
                    rhs_matrix[e_k.dofs[1, i], 0] = interior_val[1]
            continue

        # Add the mass matrix term to the right hand side.
        _element_mass(mass_matrix, e_k)
        for e_l in mesh:
            # Compute and add the RHS and matrix terms to the system.
            _element_pair(lhs_matrix, e_k, e_l, which_kernels, "matrix")
            _element_pair(rhs_matrix, e_k, e_l, which_kernels, "rhs")


    # Combine the two rhs terms
    rhs = np.sum(rhs_matrix, axis = 1)
    mass_rhs = np.sum(mass_matrix, axis = 1)
    rhs += mass_rhs

    # Return the fully assembled linear system
    return lhs_matrix, rhs

def _identity_matrix(matrix, e_k):
    for i in range(e_k.basis.n_fncs):
        matrix[e_k.dofs[0, i], e_k.dofs[0, i]] = 1.0
        matrix[e_k.dofs[1, i], e_k.dofs[1, i]] = 1.0

def _element_mass(matrix, e_k):
    # Because the term is identical (just replace u by t) for both
    # integral equations, this function does not care about the BC type
    bc_basis = e_k.bc.basis
    if type(bc_basis) is ZeroBasis:
        return

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
    if type(e_k_basis) is ZeroBasis:
        return

    # Determine what quadrature formula to use
    quad_outer, quad_inner = e_k.qs.get_quadrature(
                            kernel.singularity_type, e_k, e_l)


    # Handle point sources
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
        e_l_pt_srcs = []

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

    if type(e_l_basis) is ZeroBasis:
        return

    # Handle the integration of pairs of basis functions
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

