
def _sgbem_interior_matrix(mesh, pt, normal, kernel_set, type):
    for e_l in mesh:
        if type(e_l.basis) is ZeroBasis:
            return
        kernel, factor = interior_which_kernels[type][e_l.bc.type]
        e_l_basis = _choose_basis(e_l.basis, kernel.soln_gradient)
        assert(kernel.test_gradient == False)

# def _disp_disc_element_pair(rhs_matrix, e_k, e_l, kernel_set):
#     e_l_basis, e_l_pt_srcs =
#         _choose_basis(e_l.bc.basis, kernel.soln_gradient)
#     for i in range(e_k.basis.n_fncs):
#         for e_l_pt in e_l_pt_srcs:
#             e_l_dof = e_l_pt[1]
#             phys_pt_l = e_l.mapping.get_physical_point(e_l_pt[0][0])
#             phys_pt_k = e_k.mapping.get_physical_point(e_k.basis.nodes[i])
#             normal_l = e_l.mapping.get_normal(e_l_pt[0][0])
#             strength = [e_l_pt[0][1], e_l_pt[0][2]]
#             traction = kernel_set.k_sh.call(phys_pt_l - phys_pt_k
#             for idx1 in range(2):
#                 for idx2 in range(2):
#                     matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, e_l_dof]] += \
#                         integral[idx1][idx2]
