import numpy as np
from codim1.core import *
from codim1.fast_lib import HypersingularKernel, AdjointTractionKernel
from codim1.assembly.interior_point import interior_pt_rhs,\
                                           interior_pt_soln

def setup():
    msh = circular_mesh(10, 1.0)
    bf = basis_funcs.BasisFunctions.from_degree(0)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    init_dofs(msh)
    qs = QuadStrategy(msh, 6, 12, 2, 2)
    return msh, qs


# Some ugly but real world tests. Couldn't think of anything simpler to try it
# on....
def test_interior_point_hypersingular():
    msh, qs = setup()
    coeffs = np.array([ -1.51048858e-02,  -6.11343409e-03,  -1.97679606e-11,
         6.11343405e-03,   1.51048857e-02,   1.51048857e-02,
         6.11343405e-03,  -1.97684512e-11,  -6.11343409e-03,
        -1.51048858e-02,   3.29088618e-03,   1.41159450e-02,
         4.68710442e-02,   1.41159450e-02,   3.29088618e-03,
        -3.29088619e-03,  -1.41159451e-02,  -4.68710442e-02,
        -1.41159451e-02,  -3.29088619e-03])
    kernel = HypersingularKernel(1.0, 0.25)
    pts_normals = (np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    result = interior_pt_soln(msh, qs, pts_normals, kernel, coeffs)
    np.testing.assert_almost_equal(result[0], -0.01064342343)
    np.testing.assert_almost_equal(result[1], 0.0)

def test_interior_point_traction_adjoint():
    msh, qs = setup()
    def section_traction(x, d):
        if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
            x_length = np.sqrt(x[0] ** 2 + x[1] ** 2)
            return -x[d] / x_length
        return 0.0
    traction_function = basis_funcs.BasisFunctions.\
            from_function(section_traction)
    kernel = AdjointTractionKernel(1.0, 0.25)
    pts_normals = (np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    result = interior_pt_rhs(msh, qs,
                            pts_normals, kernel, traction_function)
    np.testing.assert_almost_equal(result[0], -0.002094,4)
    np.testing.assert_almost_equal(result[1], 0.0)

if __name__ == "__main__":
    test_interior_point_traction_adjoint()
