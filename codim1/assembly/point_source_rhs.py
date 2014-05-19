import numpy as np
from codim1.fast_lib import single_integral, ConstantEval
from codim1.core.quadrature import QuadSingularTelles

def point_source_rhs(mesh, qs, str_loc_norm, kernel):
    """
    Creates RHS terms that consist of a point source. Depending on the
    kernel chosen, these point sources could be:
    1. point forces
    2. point displacements
    3. point displacement discontinuity gradients
    """
    total_dofs = mesh.total_dofs
    rhs = np.zeros(total_dofs)
    for (str, loc, normal) in str_loc_norm:
        strength = ConstantEval(np.array(str))
        kernel.set_interior_data(np.array(loc), np.array(normal))
        for e_k in mesh:
            quad_info = qs.get_point_source_quadrature(
                    kernel.singularity_type, loc, e_k).quad_info
            for i in range(e_k.basis.n_fncs):
                integral = single_integral(e_k.mapping.eval,
                                       kernel,
                                       e_k.basis._basis_eval,
                                       strength,
                                       quad_info,
                                       i, 0)
                for idx1 in range(2):
                    for idx2 in range(2):
                        rhs[e_k.dofs[idx1, i]] += integral[idx1][idx2]
    return rhs
