import numpy as np
from codim1.assembly import simple_rhs_assemble
from codim1.fast_lib import TractionKernel
from codim1.core import *
from codim1.core.tools import interpolate

# A presumed-to-be correct matrix formed from the G_up kernel
# This was computed using an old formulation where the RHS was not
# interpolated first
correct_matrix = \
    np.array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -3.82120741e-04,   4.22000426e-02,   1.08527120e-02],
   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -4.21978289e-02,   3.60822483e-16,   4.21978289e-02],
   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
     -1.08527120e-02,  -4.22000426e-02,   3.82120741e-04],
   [  3.82120741e-04,  -4.22000426e-02,  -1.08527120e-02,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  4.21978289e-02,  -3.60822483e-16,  -4.21978289e-02,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  1.08527120e-02,   4.22000426e-02,  -3.82120741e-04,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])

def rhs_assembler():
    bf = basis_from_degree(1)
    msh = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    qs = QuadStrategy(msh, 10, 10, 10, 10)
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    apply_to_elements(msh, "qs", qs, non_gen = True)
    init_dofs(msh)
    return msh

def test_rhs():
    msh = rhs_assembler()
    kernel = TractionKernel(1.0, 0.25)
    # If we sum across rows, we should get the RHS value for
    # a function that is 1 everywhere
    rhs_correct = np.sum(correct_matrix, axis = 0)

    f = lambda x, n: [1.0, 1.0]
    apply_coeffs(msh, interpolate(f, msh), "rhs_f")
    # Make the function look like a basis function. It is one! The only one!
    rhs = simple_rhs_assemble(msh, lambda e: e.rhs_f, kernel)
    np.testing.assert_almost_equal(rhs_correct, rhs, 3)


