
class InteriorPoint(object):
    """
    Compute the value of the solution at an interior point.

    A lot of this code is exactly what's found in the Assembler class. Seems
    like an opportunity for some better abstraction.
    """
    def __init__(self,
                 mesh,
                 basis_funcs,
                 kernel,
                 dof_handler,
                 quadrature):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.kernel = kernel
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy

    def compute(self, displacements, tractions):
        pass
