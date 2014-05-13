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
                 src_basis_funcs,
                 soln_basis_funcs,
                 dof_handler,
                 quadrature,
                 compute_on_init = False):
        self.mesh = mesh
        self.src_basis_funcs = src_basis_funcs
        self.soln_basis_funcs = soln_basis_funcs
        self.dof_handler = dof_handler
        self.quadrature = quadrature
        self.computed = False
        if compute_on_init:
            self.compute()

    def compute(self):
        if self.computed:
            return

        total_dofs = self.dof_handler.total_dofs
        self.M = np.zeros((total_dofs, total_dofs))
        kernel = MassMatrixKernel(0, 0)
        q_info = self.quadrature.quad_info
        for k in range(self.mesh.n_elements):
            e_k = self.mesh.elements[k]
            for i in range(self.src_basis_funcs.num_fncs):
                i_dof_x = self.dof_handler.dof_map[0, k, i]
                i_dof_y = self.dof_handler.dof_map[1, k, i]
                for j in range(self.soln_basis_funcs.num_fncs):
                    j_dof_x = self.dof_handler.dof_map[0, k, j]
                    j_dof_y = self.dof_handler.dof_map[1, k, j]
                    M_local = single_integral(e_k.mapping.eval,
                                      kernel,
                                      self.src_basis_funcs._basis_eval,
                                      self.soln_basis_funcs._basis_eval,
                                      q_info,
                                      k, i, j)
                    self.M[i_dof_x, j_dof_x] += M_local[0][0]
                    self.M[i_dof_y, j_dof_y] += M_local[1][1]
        self.computed = True

    def for_rhs(self):
        return np.sum(self.M, axis = 1)
