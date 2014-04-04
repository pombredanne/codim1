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
                 basis_funcs,
                 dof_handler,
                 quad_strategy):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy

    def assemble_rhs(self, fnc, kernel):
        total_dofs = self.dof_handler.total_dofs
        rhs = np.zeros(total_dofs)
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dof_x = self.dof_handler.dof_map[0, k, i]
                dof_y = self.dof_handler.dof_map[1, k, i]
                row_x, row_y = self.assemble_row(fnc, kernel, k, i)
                rhs[dof_x] += row_x
                rhs[dof_y] += row_y
        return rhs

    def assemble_row(self, fnc, kernel, k, i):
        row_x = 0.0
        row_y = 0.0
        for l in range(self.mesh.n_elements):
            (quad_outer, quad_inner) = self.quad_strategy.get_quadrature(
                                            kernel.singularity_type, k, l)
            quad_outer_info = quad_outer.quad_info
            quad_inner_info = [q.quad_info for q in quad_inner]
            value = fl.double_integral(
                            self.mesh.mesh_eval,
                            self.mesh.is_linear,
                            kernel,
                            self.basis_funcs._basis_eval,
                            fnc._basis_eval,
                            quad_outer_info, quad_inner_info,
                            k, i, l, 0)
            row_x += value[0][0]
            row_x += value[0][1]
            row_y += value[1][0]
            row_y += value[1][1]
        return row_x, row_y
