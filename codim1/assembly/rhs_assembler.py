import numpy as np
import codim1.fast_lib as fl

"""
Performs the same basic task that simple_matrix_assemble does, however
the solution is already known as a function defined everywhere in space.
Hence, we do not decompose the function into its basis. Instead, we just
integrate it!

Assemble the rhs vector corresponding to:
\int_{\Gamma_x}\int_{\Gamma_y}\phi_i(x)*K(y,x)dy f(x) dx
"""
def simple_rhs_assemble(mesh, qs, fnc, kernel):
    rhs = np.zeros(mesh.total_dofs)
    for e_k in mesh:
        for i in range(e_k.basis.n_fncs):
            for e_l in mesh:
                (quad_outer, quad_inner) = qs.get_quadrature(
                                            kernel.singularity_type, e_k, e_l)
                quad_outer_info = quad_outer.quad_info
                quad_inner_info = [q.quad_info for q in quad_inner]
                value = fl.double_integral(
                                e_k.mapping.eval,
                                e_l.mapping.eval,
                                kernel,
                                e_k.basis._basis_eval,
                                fnc._basis_eval,
                                quad_outer_info, quad_inner_info,
                                i, 0)
                rhs[e_k.dofs[0, i]] += value[0][0]
                rhs[e_k.dofs[0, i]] += value[0][1]
                rhs[e_k.dofs[1, i]] += value[1][0]
                rhs[e_k.dofs[1, i]] += value[1][1]
    return rhs
