import numpy as np
from codim1.core import *
from codim1.assembly import *
from codim1.fast_lib import RegularizedHypersingularKernel,\
        ConstantEval, single_integral

def test_displacement_discontinuity_derivative():
    bf = BasisFunctions.from_degree(1)
    msh = simple_line_mesh(2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    init_dofs(msh)

    qs = QuadStrategy(msh, 10, 10, 10, 10)
    k_rh = RegularizedHypersingularKernel(1.0, 0.25)

    # basis function should be equal to x_hat
    k = 1
    i = 1
    qi = qs.get_simple().quad_info
    strength = ConstantEval([1.0, 1.0])
    k_rh.set_interior_data(np.array([-2.0, 0.0]), np.array([0.0, 1.0]))
    basis = bf.get_gradient_basis(msh)._basis_eval
    result = single_integral(msh.elements[k].mapping.eval, k_rh, strength,
                    basis, qi, 0, i)
    np.testing.assert_almost_equal(result[1][1], 0.193011, 4)
    np.testing.assert_almost_equal(result[0][0], -0.0191957, 4)

    i = 0
    result = single_integral(msh.elements[k].mapping.eval, k_rh, strength,
                    basis, qi, 0, i)
    np.testing.assert_almost_equal(result[1][1], -0.193011, 4)
    np.testing.assert_almost_equal(result[0][0], 0.0191957, 4)

    strlocnorm = [((1.0, 0.0), np.array([-2.0, 0.0]), np.zeros(2))]
    rhs = point_source_rhs(msh, qs, strlocnorm, k_rh)

if __name__ == "__main__":
    test_displacement_discontinuity_derivative()
