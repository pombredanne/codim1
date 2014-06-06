import numpy as np
import codim1.fast_lib as fl

"""
Performs the same basic task that simple_matrix_assemble does, however
the assembles a rhs vector instead of a matrix.

Assemble the rhs vector corresponding to:
\int_{\Gamma_x}\int_{\Gamma_y}\phi_i(x)*K(y,x)dy f(x) dx
"""

def simple_rhs_assemble(mesh, f_grabber, kernel):
    rhs = np.zeros(mesh.total_dofs)
    for e_k in mesh:
        rhs_k_basis = (e_k.basis if not kernel.test_gradient
                       else e_k.basis.get_gradient_basis())
        for i in range(e_k.basis.n_fncs):
            for e_l in mesh:
                rhs_l_basis = f_grabber(e_l)
                rhs_l_basis = (rhs_l_basis if not kernel.soln_gradient
                               else rhs_l_basis.get_gradient_basis())
                (quad_outer, quad_inner) = e_k.qs.get_quadrature(
                                            kernel.singularity_type, e_k, e_l)
                for j in range(e_l.basis.n_fncs):
                    value = fl.double_integral(
                                    e_k.mapping.eval,
                                    e_l.mapping.eval,
                                    kernel,
                                    rhs_k_basis,
                                    rhs_l_basis,
                                    quad_outer, quad_inner,
                                    i, j)
                    rhs[e_k.dofs[0, i]] += value[0][0]
                    rhs[e_k.dofs[0, i]] += value[0][1]
                    rhs[e_k.dofs[1, i]] += value[1][0]
                    rhs[e_k.dofs[1, i]] += value[1][1]
    return rhs
