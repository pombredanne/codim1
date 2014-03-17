import numpy as np

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
                 basis_funcs,
                 dof_handler,
                 quadrature,
                 compute_on_init = False):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.dof_handler = dof_handler
        self.quadrature = quadrature
        self.computed = False
        if compute_on_init:
            self.compute()

    def compute(self):
        if self.computed:
            return

        # TODO: use the scipy sparse matrices here.
        total_dofs = self.dof_handler.total_dofs
        self.M = np.zeros((total_dofs, total_dofs))
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                i_dof_x = self.dof_handler.dof_map[0, k, i]
                i_dof_y = self.dof_handler.dof_map[1, k, i]
                for j in range(self.basis_funcs.num_fncs):
                    j_dof_x = self.dof_handler.dof_map[0, k, j]
                    j_dof_y = self.dof_handler.dof_map[1, k, j]
                    M_local = self.single_integral(k, i, j)
                    self.M[i_dof_x, j_dof_x] = M_local[0]
                    self.M[i_dof_y, j_dof_y] = M_local[1]
        self.computed = True

    def single_integral(self, k, i, j):
        """
        Performs a single integral over the element specified by k
        with the basis functions specified by i and j.  Kernel should be
        a function that can be evaluated at all point within the element
        and (is not singular!)
        """

        # Jacobian used to transfer the integral back to physical coordinates
        jacobian = self.mesh.get_element_jacobian(k)

        # Just perform standard gauss quadrature
        q_pts = self.quadrature.x
        w = self.quadrature.w
        result = 0.0
        for (q_pt, w) in zip(q_pts, w):
            phys_pt = self.mesh.get_physical_points(k, q_pt)
            # The basis functions should be evaluated on reference
            # coordinates
            src_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt, phys_pt)
            soln_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt, phys_pt)
            result += soln_basis_fnc * src_basis_fnc * jacobian * w
        return result
