import numpy as np
from codim1.core import *
from codim1.fast_lib import HypersingularKernel, AdjointTractionKernel,\
        AlignedInteriorPoint
from codim1.assembly.interior_point import interior_pt
from codim1.core.tools import interpolate

def int_pt_test_setup(n):
    msh = circular_mesh(n, 1.0)
    bf = basis_funcs.basis_from_degree(0)
    qs = QuadStrategy(msh, 6, 12, 2, 2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    apply_to_elements(msh, "qs", qs, non_gen = True)
    init_dofs(msh)
    return msh


# Some ugly but real world tests. Couldn't think of anything simpler to try it
# on....
def test_interior_point_hypersingular():
    msh = int_pt_test_setup(10)
    coeffs = np.array([ -1.51048858e-02,  -6.11343409e-03,  -1.97679606e-11,
         6.11343405e-03,   1.51048857e-02,   1.51048857e-02,
         6.11343405e-03,  -1.97684512e-11,  -6.11343409e-03,
        -1.51048858e-02,   3.29088618e-03,   1.41159450e-02,
         4.68710442e-02,   1.41159450e-02,   3.29088618e-03,
        -3.29088619e-03,  -1.41159451e-02,  -4.68710442e-02,
        -1.41159451e-02,  -3.29088619e-03])
    apply_coeffs(msh, coeffs, "soln")

    kernel = HypersingularKernel(1.0, 0.25)
    pts_normals = (np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    result = interior_pt(msh, pts_normals, kernel)
    np.testing.assert_almost_equal(result[0], -0.01064342343)
    np.testing.assert_almost_equal(result[1], 0.0)

def test_interior_point_traction_adjoint():
    msh = int_pt_test_setup(50)
    def section_traction(x, n):
        if np.abs(x[0]) < np.cos(24 * (np.pi / 50)):
            x_length = np.sqrt(x[0] ** 2 + x[1] ** 2)
            return (-x[0] / x_length, -x[1] / x_length)
        return (0.0, 0.0)
    traction_coeffs = interpolate(section_traction, msh)
    apply_coeffs(msh, traction_coeffs, "trac_fnc")
    kernel = AdjointTractionKernel(1.0, 0.25)
    pts_normals = (np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    result = interior_pt(msh, pts_normals, kernel,
                             lambda e: e.trac_fnc)
    np.testing.assert_almost_equal(result[0], -0.002094,4)
    np.testing.assert_almost_equal(result[1], 0.0)

def test_gll_interior_point():
    # Setup a mesh with a GLL basis
    msh = circular_mesh(1, 1.0)
    bf = basis_funcs.basis_from_degree(9)
    qs = GLLQuadStrategy(msh, 10, 12, 2, 2)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", False, non_gen = True)
    apply_to_elements(msh, "qs", qs, non_gen = True)
    init_dofs(msh)

    coeffs = np.array([ -1.51048858e-02,  -6.11343409e-03,  -1.97679606e-11,
         6.11343405e-03,   1.51048857e-02,   1.51048857e-02,
         6.11343405e-03,  -1.97684512e-11,  -6.11343409e-03,
        -1.51048858e-02,   3.29088618e-03,   1.41159450e-02,
         4.68710442e-02,   1.41159450e-02,   3.29088618e-03,
        -3.29088619e-03,  -1.41159451e-02,  -4.68710442e-02,
        -1.41159451e-02,  -3.29088619e-03])
    apply_coeffs(msh, coeffs, "soln")

    kernel = HypersingularKernel(1.0, 0.25)
    pts_normals = (np.array([0.5, 0.0]), np.array([1.0, 0.0]))
    result = interior_pt(msh, pts_normals, kernel)
    result2 = interior_pt(msh, pts_normals, kernel,
                          integrator = AlignedInteriorPoint)
    np.testing.assert_almost_equal(result, result2)

if __name__ == "__main__":
    test_interior_point_traction_adjoint()
