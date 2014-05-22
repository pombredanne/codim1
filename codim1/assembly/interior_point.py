import numpy as np
from codim1.core import QuadGauss
from codim1.fast_lib import single_integral, ConstantBasis

def interior_pt_rhs(mesh, pt_normal, kernel, rhs_fnc):
    result = np.zeros(2)
    kernel.set_interior_data(pt_normal[0], pt_normal[1])
    one = ConstantBasis(np.ones(2))
    for e_k in mesh:
        quad_info =\
            e_k.qs.get_interior_quadrature(e_k, pt_normal[0]).quad_info

        integral = single_integral(e_k.mapping.eval,
                                   kernel,
                                   one,
                                   rhs_fnc,
                                   quad_info,
                                   0, 0)
        for idx1 in range(2):
            for idx2 in range(2):
                result[idx1] += integral[idx1][idx2]
    return result

def interior_pt_soln(mesh, pt_normal, kernel, coeffs):
    result = np.zeros(2)
    kernel.set_interior_data(pt_normal[0], pt_normal[1])
    one = ConstantBasis(np.ones(2))
    for e_k in mesh:
        quad_info =\
            e_k.qs.get_interior_quadrature(e_k, pt_normal[0]).quad_info

        for i in range(e_k.basis.n_fncs):
            dof = e_k.dofs[:, i]
            integral = single_integral(e_k.mapping.eval,
                                       kernel,
                                       one,
                                       e_k.basis,
                                       quad_info,
                                       0, i)
            for idx1 in range(2):
                for idx2 in range(2):
                    result[idx1] += integral[idx1][idx2] * coeffs[dof[idx2]]
    return result
