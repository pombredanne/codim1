import numpy as np
from codim1.fast_lib import double_integral
"""
This function computes the kernel function matrices needed by a boundary
element method. These matrices are of the form:
\int_{Gamma_i}\int_{\Gamma_j} K(x, y) \phi_i(x) \phi_j(y) dx dy

Note that if K is to be interpreted in a cauchy principal value sense,
two things need to be done:
1. Use the proper quadrature formula. The Piessens quadrature works well
for things like 1/r.
2. Account for the contribution of the singularity to the integral. This
will normally take the form of adding or subtracting 0.5 * M where M
is the mass matrix. This step is not performed by this class and needs
to be done independently. Talk to your math to figure this out.

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

def simple_matrix_assemble(mesh, qs, kernel):
    total_dofs = mesh.total_dofs
    matrix = np.zeros((total_dofs, total_dofs))
    for e_k in mesh:
        for e_l in mesh:
            _compute_element_pair(matrix, e_k, e_l, qs, kernel)
    return matrix

def _compute_element_pair(matrix, e_k, e_l, qs, kernel):
    for i in range(e_k.basis.n_fncs):
        for j in range(e_k.basis.n_fncs):
            integral = _compute_one_interaction(qs, kernel, e_k, i, e_l, j)
            for idx1 in range(2):
                for idx2 in range(2):
                    matrix[e_k.dofs[idx1, i], e_l.dofs[idx2, j]] += \
                        integral[idx1][idx2]

def _compute_one_interaction(qs, kernel, e_k, i, e_l, j):
    (quad_outer, quad_inner) = qs.get_quadrature(
                            kernel.singularity_type, e_k, e_l)
    quad_outer_info = quad_outer.quad_info
    quad_inner_info = [q.quad_info for q in quad_inner]
    integral = double_integral(
                    e_k.mapping.eval,
                    e_l.mapping.eval,
                    kernel,
                    e_k.basis._basis_eval,
                    e_l.basis._basis_eval,
                    quad_outer_info, quad_inner_info,
                    i, j)
    return integral
