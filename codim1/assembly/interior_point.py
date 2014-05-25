import numpy as np
from codim1.fast_lib import single_integral, ConstantBasis


def interior_pt_rhs(mesh, pt_normal, kernel, rhs_fnc):
    """
    Calculate the influence at an interior point from a boundary condition
    """
    result = np.zeros(2)
    kernel.set_interior_data(pt_normal[0], pt_normal[1])
    for e_k in mesh:
        quad_info, interior_integrator = \
            e_k.qs.get_interior_quadrature(e_k, pt_normal[0])

        integral = interior_integrator(e_k.mapping.eval, kernel,
                                       rhs_fnc, quad_info, 0)
        for idx1 in range(2):
            for idx2 in range(2):
                result[idx1] += integral[idx1][idx2]
    return result

def interior_pt_soln(mesh, pt_normal, kernel, coeffs):
    """
    Calculate the influence at an interior point from the solution.
    """
    result = np.zeros(2)
    kernel.set_interior_data(pt_normal[0], pt_normal[1])
    for e_k in mesh:
        quad_info, interior_integrator = \
            e_k.qs.get_interior_quadrature(e_k, pt_normal[0])

        for i in range(e_k.basis.n_fncs):
            dof = e_k.dofs[:, i]
            integral = interior_integrator(e_k.mapping.eval, kernel,
                                           e_k.basis, quad_info, i)
            for idx1 in range(2):
                for idx2 in range(2):
                    result[idx1] += integral[idx1][idx2] * coeffs[dof[idx2]]
    return result
