import numpy as np
import codim1.fast_lib as fl

class RHSAssembler(object):
    """
    Performs the same basic task that MatrixAssembler does, however
    the solution is already known as a function defined everywhere in space.
    Hence, we do not decompose the function into its basis. Instead, we just
    integrate it!

    Assemble the rhs vector corresponding to:
    \int_{\Gamma_x}\int_{\Gamma_y}\phi_i(x)*K(y,x)dy f(x) dx
    """
    def __init__(self,
                 mesh,
                 quad_strategy):
        self.mesh = mesh
        self.quad_strategy = quad_strategy

    def assemble_rhs(self, fnc, kernel):
        rhs = np.zeros(self.mesh.total_dofs)
        for e_k in self.mesh:
            for i in range(self.basis_funcs.num_fncs):
                dof_x = e_k.dofs[0, i]
                dof_y = e_k.dofs[1, i]
                row_x, row_y = self.assemble_row(e_k, fnc, kernel, i)
                rhs[dof_x] += row_x
                rhs[dof_y] += row_y
        return rhs

    def assemble_row(self, e_k, fnc, kernel, i):
        row_x = 0.0
        row_y = 0.0
        for e_l in self.mesh:
            (quad_outer, quad_inner) = self.quad_strategy.get_quadrature(
                                        kernel.singularity_type, e_k, e_l)
            quad_outer_info = quad_outer.quad_info
            quad_inner_info = [q.quad_info for q in quad_inner]
            value = fl.double_integral(
                            e_k.mapping.eval,
                            e_l.mapping.eval,
                            kernel,
                            e_k.basis_funcs._basis_eval,
                            fnc._basis_eval,
                            quad_outer_info, quad_inner_info,
                            i, 0)
            row_x += value[0][0]
            row_x += value[0][1]
            row_y += value[1][0]
            row_y += value[1][1]
        return row_x, row_y
