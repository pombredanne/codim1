import numpy as np
import codim1.core.quadrature as quadrature
import codim1.fast.integration as integration

class Assembler(object):
    """
    This class computes the kernel function matrices needed by a boundary
    element method. These matrices are of the form:
    \int_{Gamma_i}\int_{\Gamma_j} K(x, y) \phi_i(x) \phi_j(y) dx dy

    Note that if K is to be interpreted in a cauchy principal value sense,
    two things need to be done:
    1. Use the proper quadrature formula. The Piessens quadrature works well.
    2. Account for the contribution of the singularity to the integral. This
    will normally take the form of adding or subtracting 0.5 * M where M
    is the mass matrix. This step is not performed by this class and needs
    to be done independently.

    Four different kernels are normally used.
    1. Guu -- The displacement->displacement kernel from the displacement BIE
    2. Gup -- The traction->displacement kernel from the displacement BIE
    3. Gpu -- The displacement->traction kernel from the traction BIE
    4. Gpp -- The traction->traction kernel from the traction BIE

    #1 is improperly integrable. #2, #3 are strongly singular, so they
    must be interpreted in the Cauchy principal value sense.
    #4 is hypersingular and is only useful because of its origins in a real
    physical model. It can be interpreted as a "Hadamard finite part"
    integral. But, in this code it is integrated by parts to reduce the
    singularity to be improperly integrable.
    """
    def __init__(self,
                 mesh,
                 basis_funcs,
                 dof_handler,
                 quad_strategy):
        self.mesh = mesh
        self.basis_funcs = basis_funcs
        self.dof_handler = dof_handler
        self.quad_strategy = quad_strategy

    def assemble_matrix(self, kernel, singularity_type):
        """
        Assemble a matrix representing the double integral over
        sources and solutions against the given kernel function.
        singularity_type specifies whether the kernel function is
        weakly or strongly singular ("logr" vs. "oneoverr").
        """
        total_dofs = self.dof_handler.total_dofs
        G = np.zeros((total_dofs, total_dofs))
        for k in range(self.mesh.n_elements):
            for i in range(self.basis_funcs.num_fncs):
                dof_x = self.dof_handler.dof_map[0, k, i]
                dof_y = self.dof_handler.dof_map[1, k, i]
                (G_row_x, G_row_y) = self.assemble_row(kernel,
                                            singularity_type, k, i)
                G[dof_x, :] += G_row_x
                G[dof_y, :] += G_row_y
        return G

    def assemble_row(self, kernel, singularity_type, k, i):
        G_row_x = np.zeros(self.dof_handler.total_dofs)
        G_row_y = np.zeros(self.dof_handler.total_dofs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                soln_dof_x = self.dof_handler.dof_map[0, l, j]
                soln_dof_y = self.dof_handler.dof_map[1, l, j]

                G_local = self.assemble_one_interaction(kernel,
                        singularity_type, k, i, l, j)
                # # M_local is only applied on the block diagonal
                # H_local[0, 0] += M_local
                # H_local[1, 1] += M_local

                # Add the local interactions to the global matrix in
                # the proper locations
                G_row_x[soln_dof_x] += G_local[0, 0]
                G_row_x[soln_dof_y] += G_local[0, 1]
                G_row_y[soln_dof_x] += G_local[1, 0]
                G_row_y[soln_dof_y] += G_local[1, 1]
        return (G_row_x, G_row_y)

    def assemble_one_interaction(self, kernel, singularity_type, k, i, l, j):
        """
        Compute one pair of element interactions
        """
        (G_quad_outer, G_quad_inner) = self.quad_strategy.get_quadrature(
                                            singularity_type, k, l)
        G_local = integration.double_integral(
                        self.mesh,
                        self.basis_funcs,
                        kernel,
                        G_quad_outer,
                        G_quad_inner,
                        k, i, l, j)
        return G_local

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
