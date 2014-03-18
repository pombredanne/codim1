import numpy as np
import codim1.fast.integration as integration

class MatrixAssembler(object):
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

    Two boundary integral equations are used:
    DBIE -- The variable of interest is the displacement. All integrals have
    units of displacement.
    TBIE -- Normal derivative of the DBIE. The variable of interest is the
    traction. All integrals have units of traction.

    Four different kernels are normally used.
    1. Guu -- Represents the test tractions and is multiplied by the solution
              displacements for the DBIE
    2. Gup -- Represents the test displacements and is multiplied by the
              solution tractions for the DBIE
    3. Gpu -- Represents the test displacements and is multiplied by the
              solution tractions for the TBIE
    4. Gpp -- Represents the test tractions and is multiplied by the solution
              displacements for the TBIE

    #1 is improperly integrable. #2, #3 are strongly singular, so they
    must be interpreted in the Cauchy principal value sense.
    #4 is hypersingular and is only useful because of its origins in a real
    physical model (in other words, the hypersingularity cancels out,
    otherwise, reality would be infinite!).
    It can be interpreted as a "Hadamard finite part"
    integral, by separating out the divergent (infinite) terms from the
    convergent ones. However, in this code it is integrated by parts to
    reduce the singularity to be improperly integrable -- a much easier
    solution.
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

    def assemble_matrix(self, kernel):
        """
        Assemble a matrix representing the double integral over
        sources and solutions against the given kernel function.
        singularity_type specifies whether the kernel function is
        weakly or strongly singular ("logr" vs. "oneoverr").
        """
        total_dofs = self.dof_handler.total_dofs
        G = np.zeros((total_dofs, total_dofs))
        for k in range(self.mesh.n_elements):
            if k % 25 == 0:
                print "Assembling element " + str(k)
            for i in range(self.basis_funcs.num_fncs):
                dof_x = self.dof_handler.dof_map[0, k, i]
                dof_y = self.dof_handler.dof_map[1, k, i]
                (G_row_x, G_row_y) = self.assemble_row(kernel, k, i)
                G[dof_x, :] += G_row_x
                G[dof_y, :] += G_row_y
        return G

    def assemble_row(self, kernel, k, i):
        G_row_x = np.zeros(self.dof_handler.total_dofs)
        G_row_y = np.zeros(self.dof_handler.total_dofs)
        for l in range(self.mesh.n_elements):
            for j in range(self.basis_funcs.num_fncs):
                soln_dof_x = self.dof_handler.dof_map[0, l, j]
                soln_dof_y = self.dof_handler.dof_map[1, l, j]

                G_local = self.assemble_one_interaction(kernel, k, i, l, j)
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

    def assemble_one_interaction(self, kernel, k, i, l, j):
        """
        Compute one pair of element interactions
        """
        (G_quad_outer, G_quad_inner) = self.quad_strategy.get_quadrature(
                                            kernel.singularity_type, k, l)
        G_local = integration.double_integral(
                        self.mesh,
                        kernel,
                        self.basis_funcs,
                        self.basis_funcs,
                        G_quad_outer,
                        G_quad_inner,
                        k, i, l, j)
        return G_local

    def double_integral(self, kernel, inner_quadrature, k, i, l, j):
        """Thin wrapper around the integration.double_integral method"""
        return integration.double_integral(
                        self.mesh,
                        kernel,
                        self.basis_funcs,
                        self.basis_funcs,
                        self.quad_strategy.get_simple(),
                        inner_quadrature,
                        k, i, l, j)
