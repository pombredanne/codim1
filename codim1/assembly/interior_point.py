import numpy as np
from codim1.core.basis_funcs import Function
from codim1.core import QuadGauss
from codim1.fast_lib import single_integral, ConstantEval

def interior_pt_rhs(mesh, qs, pts_normals, kernel, rhs_fnc):
    result = np.zeros(2)
    kernel.set_interior_data(pts_normals[0], pts_normals[1])
    one = ConstantEval(np.ones(2))
    for e_k in mesh:
        quad_info =\
            qs.get_interior_quadrature(e_k, pts_normals[0]).quad_info

        integral = single_integral(e_k.mapping.eval,
                                   kernel,
                                   one,
                                   rhs_fnc._basis_eval,
                                   quad_info,
                                   0, 0)
        for idx1 in range(2):
            for idx2 in range(2):
                result[idx1] += integral[idx1][idx2]
    return result

def interior_pt_soln(mesh, qs, pts_normals, kernel, coeffs):
    result = np.zeros(2)
    kernel.set_interior_data(pts_normals[0], pts_normals[1])
    one = ConstantEval(np.ones(2))
    for e_k in mesh:
        quad_info =\
            qs.get_interior_quadrature(e_k, pts_normals[0]).quad_info

        for i in range(e_k.basis.n_fncs):
            dof = e_k.dofs[:, i]
            integral = single_integral(e_k.mapping.eval,
                                       kernel,
                                       one,
                                       e_k.basis._basis_eval,
                                       quad_info,
                                       0, i)
            for idx1 in range(2):
                for idx2 in range(2):
                    result[idx1] += integral[idx1][idx2] * coeffs[dof[idx2]]
    return result
