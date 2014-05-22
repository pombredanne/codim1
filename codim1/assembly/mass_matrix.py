import numpy as np
from codim1.fast_lib import single_integral, MassMatrixKernel

"""
This function produces a classical finite element style mass matrix for
the surface basis functions.
This is a sparse matrix where each entry is an integral:
\int_{\Gamma} \phi_i \phi_j dS
This matrix is added to the kernel matrices to account for the
cauchy singularity term that arises when some kernels' integrals
are taken to the boundary. See, for example, the first term in
equations 97 and 98 in Bonnet 1998 -- SGBEM.
"""
def assemble_mass_matrix(mesh, quadrature,
        basis_grabber = lambda e: e.basis):
    total_dofs = mesh.total_dofs
    M = np.zeros((total_dofs, total_dofs))
    kernel = MassMatrixKernel(0, 0)
    for e_k in mesh:
        rhs_basis = basis_grabber(e_k)
        for i in range(e_k.basis.n_fncs):
            for j in range(rhs_basis.n_fncs):
                M_local = single_integral(e_k.mapping.eval,
                                  kernel,
                                  e_k.basis,
                                  rhs_basis,
                                  quadrature,
                                  i, j)
                # Don't take the off diagonal components because we don't
                # want to couple the x and y dimensions in the mass matrix
                M[e_k.dofs[0,i], e_k.dofs[0,j]] += M_local[0][0]
                M[e_k.dofs[1,i], e_k.dofs[1,j]] += M_local[1][1]
    return M

def mass_matrix_for_rhs(matrix):
    return np.sum(matrix, axis = 1)
