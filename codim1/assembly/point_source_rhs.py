import numpy as np
from codim1.fast_lib import single_integral, ConstantEval
from codim1.core.quadrature import QuadSingularTelles

class PointSourceRHS(object):
    """
    Creates RHS terms that consist of a point source. Depending on the
    kernel chosen, these point sources could be:
    1. point forces
    2. point displacements
    3. point displacement discontinuity gradients
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

    def assemble_rhs(self, strength_and_location, kernel):
        total_dofs = self.dof_handler.total_dofs
        rhs = np.zeros(total_dofs)
        for (str, loc, normal) in strength_and_location:
            strength = ConstantEval(np.array(str))
            kernel.set_interior_data(np.array(loc), np.array(normal))
            for k in range(self.mesh.n_elements):
                quadrature = \
                    self.quad_strategy.get_point_source_quadrature(
                            kernel.singularity_type, loc, k)
                quad_info = quadrature.quad_info
                for i in range(self.basis_funcs.num_fncs):
                    dof_x = self.dof_handler.dof_map[0, k, i]
                    dof_y = self.dof_handler.dof_map[1, k, i]
                    integral = single_integral(self.mesh.mesh_eval,
                                               self.mesh.is_linear,
                                               kernel,
                                               self.basis_funcs._basis_eval,
                                               strength,
                                               quad_info,
                                               k, i, 0)
                    rhs[dof_x] += integral[0][0]
                    rhs[dof_x] += integral[0][1]
                    rhs[dof_y] += integral[1][0]
                    rhs[dof_y] += integral[1][1]
        return rhs
