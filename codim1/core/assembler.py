import numpy as np
import codim1.core.quadrature as quadrature
import codim1.fast.integration as integration

class Assembler(object):
    """
    This class builds the matrices needed for a boundary element method.
    Two matrices are created. The terminology here is in reference to the
    derivation based on the reciprocal theorem. The definitions of these
    matrices can be derived by performing integration by parts on the
    original governing equations of static elasticity.

    1. H -- The traction kernel matrix represents the test tractions acting
    through the solution displacements at each node of each element.

    2. G -- The displacement kernel matrix represents the test displacements
    acting through the solution tractions at each node of each element.
    """
    def __init__(self,
                 mesh,
                 basis_funcs,
                 kernel,
                 dof_handler,
                 quad_strategy):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.kernel = kernel
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy


    def assemble(self):
        """
        Produces the H and G matrices as described in the class description.
        Return (H, G)
        """
        total_dofs = self.dof_handler.total_dofs
        H = np.zeros((total_dofs, total_dofs))
        G = np.zeros((total_dofs, total_dofs))
        for k in range(self.mesh.n_elements):
            print "Assembling element: " + str(k)
            for i in range(self.basis_funcs.num_fncs):
                dof_x = self.dof_handler.dof_map[0, k, i]
                dof_y = self.dof_handler.dof_map[1, k, i]

                (G_row_x, G_row_y), (H_row_x, H_row_y) = \
                    self.assemble_row(k, i)
                G[dof_x, :] += G_row_x
                G[dof_y, :] += G_row_y

                H[dof_x, :] += H_row_x
                H[dof_y, :] += H_row_y

        # Enforce the symmetry of G. Inefficient way of doing this, but it
        # works. The asymmetry only results from small numerical errors in the
        # computation, so this statement shouldn't make the error any worse.
        assert(((G - G.T) / np.max(G) < 1e-4).all())
        G -= 0.5 * (G - G.T)

        return H, G

    def assemble_row(self, k, i):
        G_row_x = np.zeros(self.dof_handler.total_dofs)
        G_row_y = np.zeros(self.dof_handler.total_dofs)
        H_row_x = np.zeros(self.dof_handler.total_dofs)
        H_row_y = np.zeros(self.dof_handler.total_dofs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                soln_dof_x = self.dof_handler.dof_map[0, l, j]
                soln_dof_y = self.dof_handler.dof_map[1, l, j]

                G_local, H_local, M_local = \
                    self.assemble_one_interaction(k, i, l, j)
                # M_local is only applied on the block diagonal
                H_local[0, 0] += M_local
                H_local[1, 1] += M_local

                # Add the local interactions to the global matrix in
                # the proper locations
                G_row_x[soln_dof_x] += G_local[0, 0]
                G_row_x[soln_dof_y] += G_local[0, 1]
                G_row_y[soln_dof_x] += G_local[1, 0]
                G_row_y[soln_dof_y] += G_local[1, 1]

                H_row_x[soln_dof_x] += H_local[0, 0]
                H_row_x[soln_dof_y] += H_local[0, 1]
                H_row_y[soln_dof_x] += H_local[1, 0]
                H_row_y[soln_dof_y] += H_local[1, 1]
        return (G_row_x, G_row_y), (H_row_x, H_row_y)

    def assemble_one_interaction(self, k, i, l, j):
        """
        Compute one pair of element interactions
        """
        (G_quad_outer, G_quad_inner), \
        (H_quad_outer, H_quad_inner) = self.quad_strategy.get_quadrature(k, l)

        G_local = integration.double_integral(
                        self.mesh,
                        self.basis_funcs,
                        self.kernel.displacement_kernel,
                        G_quad_outer,
                        G_quad_inner,
                        k, i, l, j)

        H_local = integration.double_integral(
                        self.mesh,
                        self.basis_funcs,
                        self.kernel.traction_kernel,
                        H_quad_outer,
                        H_quad_inner,
                        k, i, l, j)

        M_local = 0.0
        if k == l:
            M_local = -0.5 * self.single_integral(lambda x: 1.0, 0.0, k, i, j)

        return G_local, H_local, M_local

    def single_integral(self, kernel, result, k, i, j):
        """
        Performs a single integral over the element specified by k
        with the basis functions specified by i and j. q_pts and w
        define the quadrature rule. Kernel should be a function that
        can be evaluated at all point within the element and (is not
        singular!)
        """
        quad = self.quad_strategy.get_simple()
        jacobian = self.mesh.get_element_jacobian(k)
        q_pts = quad.x
        w = quad.w
        # Just perform standard gauss quadrature
        for (q_pt, w) in zip(q_pts, w):
            # The basis functions should be evaluated on reference
            # coordinates
            src_basis_fnc = self.basis_funcs.evaluate_basis(i, q_pt)
            soln_basis_fnc = self.basis_funcs.evaluate_basis(j, q_pt)
            # The kernel is evaluated in physical coordinates
            phys_pt = self.mesh.get_physical_points(k, q_pt)
            result += kernel(phys_pt) * \
                soln_basis_fnc * src_basis_fnc * jacobian * w
        return result

    def double_integral(self, kernel, inner_quadrature, k, i, l, j):
        """
        Performs a double integral over a pair of elements with the
        provided quadrature rule.

        In a sense, this is the core method of any BEM implementation.

        Warning: This function modifies the "result" input.
        """
        return integration.double_integral(
                        self.mesh,
                        self.basis_funcs,
                        kernel,
                        self.quad_strategy.get_simple(),
                        inner_quadrature,
                        k, i, l, j)
