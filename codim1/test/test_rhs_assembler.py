import numpy as np
from codim1.assembly import simple_rhs_assemble
from codim1.fast_lib import TractionKernel
from codim1.core import *

# A presumed-to-be correct matrix formed from the G_up kernel
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
    bf = BasisFunctions.from_degree(1)
    msh = simple_line_mesh(2, (-1.0, 0.0), (1.0, 0.0))
    apply_to_elements(msh, "basis", bf, non_gen = True)
    apply_to_elements(msh, "continuous", True, non_gen = True)
    init_dofs(msh)
    qs = QuadStrategy(msh, 10, 10, 10, 10)
    return msh, qs

def test_rhs():
    msh, qs = rhs_assembler()
    kernel = TractionKernel(1.0, 0.25)
    # If we sum across rows, we should get the RHS value for
    # a function that is 1 everywhere
    rhs_correct = np.sum(correct_matrix, axis = 1)

    f = lambda x, d: 1.0
    # Make the function look like a basis function. It is one! The only one!
    fnc = BasisFunctions.from_function(f)
    rhs = simple_rhs_assemble(msh, qs, fnc, kernel)
    np.testing.assert_almost_equal(rhs_correct, rhs)


