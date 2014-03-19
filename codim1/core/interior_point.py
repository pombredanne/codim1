import numpy as np
from codim1.fast.integration import single_integral

class InteriorPoint(object):
    """
    Compute the value of the solution at an interior point.

    A lot of this code is exactly what's found in the Assembler class. Seems
    like an opportunity for some better abstraction.

    """
    #TODO: There are no tests for any of this. HOW TO TEST IT?
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
        G_local = self.interior_integral(
        result = np.zeros(2)
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dofs = [self.dof_handler.dof_map[0, k, i],
                        self.dof_handler.dof_map[1, k, i]]

                kernel_fnc = lambda x, n: kernel.call(x - pt, pt_normal, n)

                integral =

                G_local = self.interior_integral(self.Guu,
                                np.zeros((2, 2)),
                                point, k, i)

                H_local = self.interior_integral(self.Gup,
                                np.zeros((2, 2)),
                                point, k, i)

                for a in range(2):
                    for c in range(2):
                        result[a] -= disp[dofs[c]] * H_local[a, c]
                        result[a] += trac[dofs[c]] * G_local[a, c]

        return result

    def compute_stress(self, point, disp, trac):
        stress = np.zeros((2, 2))

        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dofs = [self.dof_handler.dof_map[0, k, i],
                        self.dof_handler.dof_map[1, k, i]]

                Dx_local = self.interior_integral(self.Dijk,
                                np.zeros((2, 2, 2)),
                                point, k, i)

                Sx_local = self.interior_integral(self.Sijk,
                                np.zeros((2, 2, 2)),
                                point, k, i)

                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            stress[a, b] -= disp[dofs[c]] * S_local[a, b, c]
                            stress[a, b] += trac[dofs[c]] * D_local[a, b, c]

        return stress

    # TODO: Combine this single integral function with the one in
    # the integration module.
    def interior_integral(self, kernel, result, pt, k, i):
        """
        Performs an integral over a surface element, computing the solution
        at an interior point. Basically, this is a collocation integral
        originating from a point force (1, 1) at the interior point
        """
        jacobian = self.mesh.get_element_jacobian(k)
        normal = self.mesh.normals[k]
        q_pts = self.quadrature.x
        w = self.quadrature.w
        # Just perform standard gauss quadrature
        for (q_pt, w) in zip(q_pts, w):
            # The kernel is evaluated in physical coordinates
            phys_soln_pt = self.mesh.get_physical_points(k, q_pt)

            # The basis functions should be evaluated on reference
            # coordinates
            soln_basis_fnc = self.basis_funcs.evaluate_basis(i,
                                    q_pt, phys_soln_pt)

            r = phys_soln_pt - pt
            result += kernel.call(r, np.zeros(2), normal) * \
                soln_basis_fnc * jacobian * w

        return result
