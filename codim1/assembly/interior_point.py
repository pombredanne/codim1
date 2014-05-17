import numpy as np
from codim1.core.basis_funcs import Function
from codim1.core import QuadGauss
from codim1.fast_lib import single_integral, ConstantEval

class InteriorPoint(object):
    """
    Computes the value of the solution at an interior point.
    """
    def __init__(self,
                 mesh,
                 dof_handler,
                 quad_strategy):
        self.mesh = mesh
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy

    def compute(self, pt, pt_normal, kernel, solution):
        """
        Determine the value of some solution at pt with normal pt_normal.
        kernel must be a standard kernel function.
        solution must behave like a set of basis functions.
        """
        result = np.zeros(2)

        kernel.set_interior_data(pt, pt_normal)
        one = ConstantEval(np.ones(2))
        for k in range(self.mesh.n_elements):
            e_k = self.mesh.elements[k]
            # Vary quadrature order depending on distance to the point.
            quadrature = self.quad_strategy.get_interior_quadrature(k, pt)
            quad_info = quadrature.quad_info
            for i in range(solution.num_fncs):
                dof = e_k[:, i]
                integral = single_integral(e_k.mapping.eval,
                                           kernel,
                                           one,
                                           e_k.basis.eval,
                                           quad_info,
                                           0, i)
                for idx1 in range(2)
                    for idx2 in range(2)
                        result[idx1] += \
                            integral[idx1][idx2] * solution[dof[idx2]]
        return result
