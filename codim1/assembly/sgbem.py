
class SGBEM(object):
    """
    This class computes the SGBEM matrix for a specified boundary value
    problem
    """
    def __init__(self,
                 k_d, k_t, k_ta, k_h,
                 mesh,
                 basis_funcs,
                 dof_handler,
                 quad_strategy,
                 displacement_bc,
                 traction_bc,
                 bc_type):
        self.k_d = k_d
        self.k_t = k_t
        self.k_ta = k_ta
        self.k_h = k_h
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy
        self.displacement_bc = displacement_bc
        self.traction_bc = traction_bc
        self.bc_type = bc_type

    def assemble_matrix(self):
        for dof_row in range(self.dof_handler.total_dofs):
            for d1:

                for dof_col in range(self.dof_handler.total_dofs):

