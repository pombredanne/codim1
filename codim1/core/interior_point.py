import numpy as np
from codim1.core.basis_funcs import Function
from codim1.fast.integration import single_integral

class InteriorPoint(object):
    """
    Computes the value of the solution at an interior point.
    """
    def __init__(self,
                 mesh,
                 dof_handler,
                 quadrature):
        self.mesh = mesh
        self.dof_handler = dof_handler
        self.quadrature = quadrature

    def compute(self, pt, pt_normal, kernel, solution):
        """
        Determine the value of some solution at pt with normal pt_normal.
        kernel must be a standard kernel function.
        solution must behave like a set of basis functions.
        """
        result = np.zeros(2)

        kernel_fnc = lambda x, n: kernel.call(x - pt, pt_normal, n)
        one = Function(lambda x: np.ones(2))
        for k in range(self.mesh.n_elements):
            for i in range(solution.num_fncs):
                integral = single_integral(self.mesh, kernel_fnc,
                                           one, solution,
                                           self.quadrature,
                                           k, 0, i)
                result[0] += integral[0, 0]
                result[0] += integral[0, 1]
                result[1] += integral[1, 0]
                result[1] += integral[1, 1]
        return result
