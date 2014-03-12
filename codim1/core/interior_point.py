import numpy as np

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
        self.quadrature = quadrature

    def compute_displacement(self, point, disp, trac):
        disp = np.zeros(2)

        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dofs = [self.dof_handler.dof_map[0, k, i],
                        self.dof_handler.dof_map[1, k, i]]

                G_local = self.interior_integral(
                                self.kernel.displacement_kernel,
                                point, k, i)

                H_local = self.interior_integral(
                                self.kernel.traction_kernel,
                                point, k, i)

                for a in range(2):
                    for c in range(2):
                        disp[a] -= disp[dofs[c]] * H_local[a, c]
                        disp[a] += trac[dofs[c]] * G_local[a, c]

        return disp

    def compute_stress(self, point, disp, trac):
        stress = np.zeros((2, 2))

        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dofs = [self.dof_handler.dof_map[0, k, i],
                        self.dof_handler.dof_map[1, k, i]]

                D_local = self.interior_integral(self.kernel.traction_adjoint,
                                point, k, i)

                S_local = self.interior_integral(self.kernel.hypersingular,
                                point, k, i)

                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            stress[a, b] -= disp[dofs[c]] * S_local[a, b, c]
                            stress[a, b] += trac[dofs[c]] * D_local[a, b, c]

        return stress

    def interior_integral(self, kernel, pt, k, i):
        """
        Performs an integral over a surface element, computing the solution
        at an interior point. Basically, this is a collocation integral
        originating from a point force (1, 1) at the interior point
        """
        jacobian = self.mesh.get_element_jacobian(k)
        normal = self.mesh.normals[k]
        q_pts = self.quadrature.x
        w = self.quadrature.w
        result = np.zeros((2, 2, 2))
        # Just perform standard gauss quadrature
        for (q_pt, w) in zip(q_pts, w):
            # The basis functions should be evaluated on reference
            # coordinates
            soln_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt)

            # The kernel is evaluated in physical coordinates
            phys_soln_pt = self.mesh.get_physical_points(k, q_pt)

            r = phys_soln_pt - pt
            result += kernel(r[0], r[1], normal[0], normal[1]) * \
                soln_basis_fnc * jacobian * w

        return result
