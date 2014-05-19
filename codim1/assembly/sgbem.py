"""
This function traverses the mesh and forms the symmetric galerkin
matrices and right hand side vectors.

This assumes that all the boundary conditions have already been attached to
the relevant element in the mesh.
"""
def sgbem_assembler(mesh, qs, kernel_set):
    total_dofs = mesh.total_dofs
    matrix = np.zeros((total_dofs, total_dofs))
    rhs = np.zeros(total_dofs)
    for e_k in mesh:
        for e_l in mesh:
            _compute_element_pair(matrix, rhs, e_k, e_l, qs, kernel_set)
    return matrix

def _compute_element_pair(matrix, rhs, e_k, e_l, qs, kernel_set):
    row_type = e_k.bc.type
    if e_k.bc.type == "disp":
        pass
    if e_k.bc.type == "trac":
        pass
    for i in range(e_k.basis.n_fncs):
        for j in range(e_k.basis.n_fncs):
            integral = compute_one_interaction(qs, kernel, e_k, i, e_l, j)
            for idx1 in range(2):
                for idx2 in range(2):
                    matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, j]] += \
                        integral[idx1][idx2]

def _compute_one_interaction(qs, kernel, e_k, i, e_l, j):
    (quad_outer, quad_inner) = qs.get_quadrature(
                            kernel.singularity_type, e_k, e_l)
    quad_outer_info = quad_outer.quad_info
    quad_inner_info = [q.quad_info for q in quad_inner]
    integral = double_integral(
                    e_k.mapping.eval,
                    e_l.mapping.eval,
                    kernel,
                    e_k.basis._basis_eval,
                    e_l.basis._basis_eval,
                    quad_outer_info, quad_inner_info,
                    i, j)
    return integral
