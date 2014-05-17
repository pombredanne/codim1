import numpy as np
from codim1.fast_lib import single_integral, MassMatrixKernel

class MassMatrix(object):
    """
    This class produces a classical finite element style mass matrix for
    the surface basis functions.
    This is a sparse matrix where each entry is an integral:
    \int_{\Gamma} \phi_i \phi_j dS
    This matrix is added to the kernel matrices to account for the
    cauchy singularity term that arises when the kernel integral
    is taken to the boundary. See, for example, the first term in
    equations 97 and 98 in Bonnet 1998 -- SGBEM.
    """
    def __init__(self,
                 mesh,
                 quadrature,
                 compute_on_init = False):
        self.mesh = mesh
        self.quadrature = quadrature
        self.computed = False
        if compute_on_init:
            self.compute()

    def compute(self):
        if self.computed:
            return

        total_dofs = self.mesh.total_dofs
        self.M = np.zeros((total_dofs, total_dofs))
        kernel = MassMatrixKernel(0, 0)
        q_info = self.quadrature.quad_info
        for e_k in self.mesh:
            for i in range(e_k.basis.n_fncs):
                i_dof_x = e_k.mesh.dofs[0, i]
                i_dof_y = e_k.mesh.dofs[1, i]
                for j in range(e_k.basis.n_fncs):
                    j_dof_x = e_k.mesh.dofs[0, j]
                    j_dof_y = e_k.mesh.dofs[1, j]
                    M_local = single_integral(e_k.mapping.eval,
                                      kernel,
                                      e_k.basis.eval,
                                      e_k.basis.eval,
                                      q_info,
                                      i, j)
                    self.M[i_dof_x, j_dof_x] += M_local[0][0]
                    self.M[i_dof_y, j_dof_y] += M_local[1][1]
        self.computed = True

    def for_rhs(self):
        return np.sum(self.M, axis = 1)
