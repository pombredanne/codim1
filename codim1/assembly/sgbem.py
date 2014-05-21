import numpy as np
from codim1.fast_lib import double_integral

"""
These functions traverse the mesh and form the symmetric galerkin
matrices and right hand side vectors.

The implementation here follows equations 14 through 17 in the Bonnet 98
paper very closely.

This assumes that all the boundary conditions have already been attached to
the relevant element in the mesh.
"""
def sgbem_assemble(mesh, qs, kernel_set):

    # Form the empty linear system
    total_dofs = mesh.total_dofs
    matrix = np.zeros((total_dofs, total_dofs))
    rhs = np.zeros(total_dofs)

    # Set the kernels for each type of boundary condition.
    which_kernels = _make_which_kernels(kernel_set)

    # Traverse the mesh and assemble the relevant terms
    for e_k in mesh:
        # Add the mass matrix term to the right hand side.
        _compute_element_mass_rhs(rhs, qs, e_k)
        for e_l in mesh:
            # Compute and add the RHS and matrix terms to the system.
            _compute_element_pair(matrix, rhs, e_k, e_l, qs, which_kernels)

    # Return the fully assembled linear system
    return matrix, rhs

def _compute_element_mass_rhs(rhs, qs, e_k):
    # Because the term is identical (just replace u by t) for both
    # integral equations, this function does not care about the BC type
    bc_basis = e_k.bc
    for i in range(e_k.basis.n_fncs):
        for j in range(rhs_basis.n_fncs):
            M_local = single_integral(e_k.mapping.eval,
                              kernel,
                              e_k.basis,
                              bc_basis,
                              q_info,
                              i, j)
            rhs[e_k.dofs[0, i]] += 0.5 * M_local[0][0]
            rhs[e_k.dofs[1, i]] += 0.5 * M_local[1][1]

def _choose_basis(element, is_gradient):
    if is_gradient:
        return element.basis
    return element.basis.get_gradient_basis()

# TODO:
# The element pair functions could be consolidated into one
# uniform interface which handles the basis function loop but not
# the basis, kernel and quadrature determination.

def _compute_element_pair_rhs(rhs, e_k, e_l, qs, which_kernels):
    # Determine which kernel and which bases to use
    rhs_kernel = which_kernels[e_k.bc.type][e_l.bc.type]["rhs"]
    rhs_k_basis = choose_basis(e_k, rhs_kernel.test_gradient)
    rhs_l_basis = choose_basis(e_l, rhs_kernel.soln_gradient)

    # Determine what quadrature formula to use
    (quad_outer, quad_inner) = qs.get_quadrature(
                            rhs_kernel.singularity_type, e_k, e_l)
    quad_outer_info = quad_outer.quad_info
    quad_inner_info = [q.quad_info for q in quad_inner]

    # Loop over basis function pairs and integrate!
    for i in range(e_k.basis.n_fncs):
        for j in range(e_k.basis.n_fncs):
            # Compute the RHS term
            # How to automate using the gradient of the boundary condition
            # when the hypersingular kernel is to be used?
            rhs_integral = double_integral(
                                e_k.mapping.eval,
                                e_l.mapping.eval,
                                kernel,
                                rhs_k_basis,
                                fnc,
                                quad_outer_info, quad_inner_info,
                                i, 0)
            for idx1 in range(2):
                for idx2 in range(2):
                    rhs[e_k.dofs[idx1, i]] += integral[idx1][idx2]

def _compute_element_pair_matrix(matrix, e_k, e_l, qs, which_kernels):
    # Determine which kernel and which bases to use
    matrix_kernel = which_kernels[e_k.bc.type][e_l.bc.type]["matrix"]
    matrix_k_basis = choose_basis(e_k, matrix_kernel.test_gradient)
    matrix_l_basis = choose_basis(e_l, matrix_kernel.soln_gradient)


    # Determine what quadrature formula to use
    (quad_outer, quad_inner) = qs.get_quadrature(
                            matrix_kernel.singularity_type, e_k, e_l)
    quad_outer_info = quad_outer.quad_info
    quad_inner_info = [q.quad_info for q in quad_inner]

    # Loop over basis function pairs and integrate!
    for i in range(e_k.basis.n_fncs):
        for j in range(e_k.basis.n_fncs):
            matrix_integral = double_integral(
                                e_k.mapping.eval,
                                e_l.mapping.eval,
                                kernel,
                                matrix_k_basis,
                                matrix_l_basis,
                                quad_outer_info, quad_inner_info,
                                i, j)

            # Insert both integrals into the global matrix and rhs
            for idx1 in range(2):
                for idx2 in range(2):
                    matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, j]] += \
                        integral[idx1][idx2]

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
            }
        }
    return which_kernels
